The goal is to predict next-period returns (y_ret_next) using XGBoost, while a Genetic Algorithm (GA) searches for a fixed-size feature subset (exactly TOP_K_FEATURES) and a good set of XGBoost hyperparameters

It runs on rolling time windows so we don’t cheat with future information.

Why ? 

Markets shift. A feature set that worked last year may not work now.

Instead of hand-picking features and params, we let a GA search combinations.

We split time forward (train → val → test) so results look like what you’d get live.

Feature engineering file is attached and  data-leakage free.

ga_xgb_rolling_featselect.py:
It loads a parquet dataset, picks numeric features (excluding date, ticker, and labels), then walks forward through the timeline in five windows; in each window it splits past data into internal train/validation, uses a GA to jointly search a fixed-size feature subset and XGBoost hyperparameters (with early stopping), selects the best individual by validation RMSE, retrains it on train∪validation, and evaluates on the next, unseen slice. Logs report split sizes, per-generation progress, and final test metrics—RMSE and directional win-rate—so you can see how well the model would have performed in a live, forward-only setting.
