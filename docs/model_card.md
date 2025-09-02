# Model Card – F1 Lap Time Predictor (Solo)
- Data: ~138k filtered race laps (pit-in removed, 60–240s suggested).
- Target: lap_time (s), residualized by season+race.
- Features: numeric + one-hots, lap_time lags(1–5), rolling means/stds, TyreLife context.
- Model: HistGradientBoostingRegressor (CPU).
- CV: GroupKFold by race (5 folds).
- Offline metrics: RMSE 3.391s, MAE 1.383s, P10–P90 coverage 73% (conformal).
- Harder regimes: WET/INTER, short stints (0–8 laps).
- Intended use: pace estimation & strategy sims; not safety-critical.
