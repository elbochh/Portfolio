"""
Expanding-window xLSTM regression + Wavelet denoise:
- For each evaluation window:
    TRAIN  = last LOOKBACK_YEARS before eval_start  (used for FS + scaler + model fit)
    VALID  = eval_start → eval_end
- Per-window feature selection on TRAIN ONLY
- Early stopping, RMSE + winrate per epoch
"""

import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ.setdefault("OMP_NUM_THREADS", "1")

import math
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# --- Feature selection backends ---
try:
    import lightgbm as lgb
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False
from sklearn.ensemble import RandomForestRegressor

# --- Optional denoise ---
try:
    import pywt
    def wavelet_denoise_1d(x, wavelet="db4", level=None):
        x = np.asarray(x, dtype=np.float32)
        finite = np.isfinite(x)
        if finite.sum() < 8:  # too short
            return x
        if not finite.all():
            m = np.nanmean(x[finite]) if finite.any() else 0.0
            x = np.where(finite, x, m)
        coeffs = pywt.wavedec(x, wavelet=wavelet, level=level)
        sigma = np.median(np.abs(coeffs[-1])) / 0.6745 + 1e-8
        uth = sigma * np.sqrt(2 * np.log(len(x)))
        den = [coeffs[0]] + [pywt.threshold(c, value=uth, mode="soft") for c in coeffs[1:]]
        return pywt.waverec(den, wavelet=wavelet)[: len(x)]
    HAS_WAVELET = True
except Exception:
    HAS_WAVELET = False

# ---------------- CONFIG (edit here) ----------------

PARQUET_PATH   = "features_lgbm_ohlcv.parquet"
OUTPUT_CSV     = "rolling_results_lookback.csv"

# Evaluation windowing
EVAL_MONTHS    = 12     # length of each validation window
STEP_MONTHS    = 12     # step between eval windows (use 1 for monthly rolling evals)
# ---------------- CONFIG (edit here) ----------------
LOOKBACK_YEARS      = 4       # model/scaler/train sequences use last 5y
FS_LOOKBACK_MONTHS  = 18      # NEW: feature selection uses only last 12 months of TRAIN


# Modeling
SEQ_LEN        = 64
TOP_K          = 31
BATCH_SIZE     = 256
EPOCHS         = 15
PATIENCE       = 3
LR             = 0.001
WEIGHT_DECAY   = 1e-4
D_MODEL        = 20
LAYERS         = 1
DROPOUT        = 0.3
DENOISE        = True
FS_BACKEND     = "lgbm"  # "lgbm", "rf", or "variance"

DEVICE = torch.device("cuda" if torch.cuda.is_available()
                      else ("mps" if torch.backends.mps.is_available() else "cpu"))
print("Device:", DEVICE)

# ------------------------ Utilities ------------------------

def normalize_colname(c: str) -> str:
    return c.strip().lower().replace(" ", "_")

def normalize_columns_df(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: normalize_colname(c) for c in df.columns})

def month_floor(d: pd.Timestamp) -> pd.Timestamp:
    d = pd.to_datetime(d)
    return pd.Timestamp(d.year, d.month, 1)

# ------------------------ Dataset ------------------------

class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self):
        return self.X.shape[0]
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ------------------------ Model ------------------------

class EMAPath(nn.Module):
    def __init__(self, d_in: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_in, d_model)
        self.gate = nn.Linear(d_in, d_model)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        z = self.proj(x)
        g = self.sigmoid(self.gate(x))
        y = []
        prev = torch.zeros_like(z[:, 0, :])
        for t in range(z.size(1)):
            prev = g[:, t, :] * z[:, t, :] + (1 - g[:, t, :]) * prev
            y.append(prev)
        return torch.stack(y, dim=1)

class XLSTMRegressor(nn.Module):
    def __init__(self, d_in: int, d_model: int = 64, layers: int = 1, dropout: float = 0.3):
        super().__init__()
        self.input_proj = nn.Linear(d_in, d_model)
        self.lstm = nn.LSTM(
            d_model, d_model, num_layers=layers, batch_first=True,
            dropout=(dropout if layers > 1 else 0.0)  # avoid PyTorch warning for 1 layer
        )
        self.ema = EMAPath(d_in, d_model)
        self.norm = nn.LayerNorm(d_model * 2)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(d_model, 1)
        )
    def forward(self, x):
        x_proj = self.input_proj(x)
        lstm_out, _ = self.lstm(x_proj)
        ema_out = self.ema(x)
        last = torch.cat([lstm_out[:, -1, :], ema_out[:, -1, :]], dim=-1)
        last = self.norm(last)
        return self.head(last).squeeze(-1)

# ------------------------ Helpers ------------------------

def build_y_date(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    df["y_date"] = df.groupby("ticker")["date"].shift(-1)
    return df.dropna(subset=["y_ret_next", "y_date"]).reset_index(drop=True)

def lgbm_feature_select(train_df, feature_cols, target_col, top_k, backend="lgbm"):
    X = train_df[feature_cols].astype(np.float32)
    y = train_df[target_col].astype(np.float32)
    if backend == "lgbm" and HAS_LGBM:
        model = lgb.LGBMRegressor(
            n_estimators=800, learning_rate=0.03,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            random_state=42, n_jobs=1
        )
        model.fit(X, y)
        imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
        return list(imp.index[:top_k])
    elif backend == "rf":
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=1)
        rf.fit(X, y)
        imp = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
        return list(imp.index[:top_k])
    else:
        return list(X.var().sort_values(ascending=False).index[:top_k])

def denoise_split(df, cols):
    if not HAS_WAVELET or not DENOISE:
        return df
    out = []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").copy()
        for c in cols:
            try:
                g[c] = wavelet_denoise_1d(g[c].to_numpy())
            except Exception:
                pass
        out.append(g)
    return pd.concat(out, axis=0, ignore_index=True)

def make_sequences(df, feature_cols, target_col, seq_len):
    X_list, y_list = [], []
    for _, g in df.groupby("ticker", sort=False):
        g = g.sort_values("date").reset_index(drop=True)
        feats = g[feature_cols].to_numpy()
        tgt = g[target_col].to_numpy()
        if len(g) <= seq_len:
            continue
        for i in range(seq_len, len(g)):
            X_list.append(feats[i - seq_len:i, :])
            y_list.append(tgt[i])
    if not X_list:
        return np.empty((0, seq_len, len(feature_cols)), np.float32), np.empty((0,), np.float32)
    return np.stack(X_list), np.asarray(y_list)

@dataclass
class Metrics:
    rmse: float
    winrate: float

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, yh = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        ys.append(yb.cpu().numpy())
        yh.append(pred.cpu().numpy())
    if not ys:
        return Metrics(float("nan"), float("nan"))
    y, yhat = np.concatenate(ys), np.concatenate(yh)
    return Metrics(math.sqrt(mean_squared_error(y, yhat)),
                   np.mean((y * yhat) > 0))

# ------------------------ Training ------------------------

def train_one_window(train_df, val_df, feature_pool, target_col, seq_len, window_tag):
    # --- NEW: slice recent TRAIN for feature selection only ---
    if FS_LOOKBACK_MONTHS is not None and FS_LOOKBACK_MONTHS > 0:
        # val_df starts at eval_start, so compute FS start by months
        fs_start = val_df["y_date"].min() - relativedelta(months=FS_LOOKBACK_MONTHS)
        fs_df = train_df[train_df["y_date"] >= fs_start].copy()
        if len(fs_df) < 1000:   # safety fallback if very small (thin history)
            fs_df = train_df
    else:
        fs_df = train_df

    # Feature selection on RECENT TRAIN only
    selected = lgbm_feature_select(fs_df, feature_pool, target_col, TOP_K, backend=FS_BACKEND)

    # Optional denoise (per split, but only on selected cols)
    train_df, val_df = denoise_split(train_df, selected), denoise_split(val_df, selected)

    # Scale with TRAIN stats only (all 5y), using the selected cols
    scaler = StandardScaler()
    scaler.fit(train_df[selected].to_numpy(np.float32))
    for df_ in (train_df, val_df):
        df_.loc[:, selected] = scaler.transform(df_[selected].to_numpy(np.float32))

    # Sequences
    Xtr, ytr = make_sequences(train_df, selected, target_col, seq_len)
    Xva, yva = make_sequences(val_df, selected, target_col, seq_len)
    if not len(Xtr) or not len(Xva):
        print(f"[{window_tag}] Skipped (not enough sequences).")
        return {"best_val_rmse": float("nan"), "best_val_winrate": float("nan")}

    tr_loader = DataLoader(SeqDataset(Xtr, ytr), batch_size=BATCH_SIZE, shuffle=True)
    va_loader = DataLoader(SeqDataset(Xva, yva), batch_size=BATCH_SIZE)

    model = XLSTMRegressor(len(selected), D_MODEL, LAYERS, DROPOUT).to(DEVICE)
    opt = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS, 1))
    loss_fn = nn.HuberLoss(delta=1.0)

    best_rmse, stale = float("inf"), 0
    best_state = None

    for ep in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

        tr, va = evaluate(model, tr_loader, DEVICE), evaluate(model, va_loader, DEVICE)
        print(f"[{window_tag}] Epoch {ep:02d} | train RMSE={tr.rmse:.5f} win={tr.winrate:.3f} | val RMSE={va.rmse:.5f} win={va.winrate:.3f}")
        if va.rmse < best_rmse:
            best_rmse, stale = va.rmse, 0
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    best = evaluate(model, va_loader, DEVICE)
    return {"best_val_rmse": best.rmse, "best_val_winrate": best.winrate}

# ------------------------ Windows ------------------------

def iter_eval_windows(df, eval_months, step_months):
    """Yield (eval_start, eval_end) month spans."""
    min_y = month_floor(df["y_date"].min())
    max_y = month_floor(df["y_date"].max())
    spans = []
    cur = min_y + relativedelta(months=eval_months)  # ensure at least one eval after some history
    while True:
        eval_start, eval_end = cur, cur + relativedelta(months=eval_months)
        if eval_start >= max_y or eval_end > max_y + relativedelta(months=1):
            break
        spans.append((eval_start, eval_end))
        cur += relativedelta(months=step_months)
    return spans

def get_train_val_split(df: pd.DataFrame, eval_start: pd.Timestamp, eval_end: pd.Timestamp,
                        lookback_years: Optional[int]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df) using rolling lookback for training."""
    if lookback_years is None:
        train_mask = df["y_date"] < eval_start
    else:
        train_start = eval_start - relativedelta(years=int(lookback_years))
        train_mask = (df["y_date"] >= train_start) & (df["y_date"] < eval_start)
    val_mask = (df["y_date"] >= eval_start) & (df["y_date"] < eval_end)
    return df.loc[train_mask].copy(), df.loc[val_mask].copy()

# ------------------------ Main ------------------------

def main():
    # Load
    df = pd.read_parquet(PARQUET_PATH)
    df = normalize_columns_df(df)
    df["date"] = pd.to_datetime(df["date"])
    df = build_y_date(df)

    # Feature pool (numeric, excluding identifiers/targets)
    exclude = {"ticker", "date", "y_date", "y_ret_next"}
    feature_pool = [c for c in df.columns if c not in exclude and pd.api.types.is_numeric_dtype(df[c])]
    print(f"Candidate features: {len(feature_pool)}")

    # Eval windows
    spans = iter_eval_windows(df, EVAL_MONTHS, STEP_MONTHS)
    print(f"Total evaluation windows: {len(spans)} | lookback={LOOKBACK_YEARS if LOOKBACK_YEARS is not None else 'ALL'} years")

    results = []
    for i, (eval_start, eval_end) in enumerate(spans, 1):
        train_df, val_df = get_train_val_split(df, eval_start, eval_end, LOOKBACK_YEARS)

        # Safety: ensure some history for sequences
        if train_df.groupby("ticker").size().max() < SEQ_LEN + 1:
            print(f"[W{i}] Skipping: not enough history for seq_len={SEQ_LEN} within lookback window.")
            continue

        tag = f"W{i} lookback:{LOOKBACK_YEARS if LOOKBACK_YEARS is not None else 'ALL'}y | train:{(eval_start - relativedelta(years=LOOKBACK_YEARS) if LOOKBACK_YEARS else '(-inf)')}→{eval_start.date()} | val:{eval_start.date()}→{eval_end.date()}"
        print("===", tag, f"(train rows={len(train_df):,}, val rows={len(val_df):,})", "===")

        res = train_one_window(train_df, val_df, feature_pool, "y_ret_next", SEQ_LEN, tag)
        results.append({
            "window": i,
            "lookback_years": LOOKBACK_YEARS if LOOKBACK_YEARS is not None else -1,
            "train_until": eval_start,
            "val_start": eval_start,
            "val_end": eval_end,
            "best_val_rmse": res["best_val_rmse"],
            "best_val_winrate": res["best_val_winrate"],
        })

    pd.DataFrame(results).to_csv(OUTPUT_CSV, index=False)
    print("Saved results to", OUTPUT_CSV)

if __name__ == "__main__":
    main()
