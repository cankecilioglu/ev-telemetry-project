# Model comparison (per-trip split)

| method | key | features | MAE_A | RMSE_A | bias_A | R2 | n_test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| training mean | baseline | - | 38.479 | 53.406 | -2.663 | -0.003 | 4835 |
| linear regression | linear | SPD, acceleration | 33.200 | 44.519 | -1.571 | 0.303 | 4835 |
| physics calibration | physics | a*v, v^3, grade, v, AIR | 27.239 | 38.792 | 0.542 | 0.471 | 4835 |
| random forest | rf | SPD, accel_mps2, ACC, BRK, ECO_bin, AIR_bin, grade_pct | 6.208 | 12.879 | -0.681 | 0.942 | 4835 |
