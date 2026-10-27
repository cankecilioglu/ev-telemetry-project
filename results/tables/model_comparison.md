# Model comparison (global chronological split)

| method | key | features | MAE_A | RMSE_A | bias_A | R2 | n_test |
| --- | --- | --- | --- | --- | --- | --- | --- |
| training mean | baseline | - | 40.348 | 54.581 | -2.835 | -0.003 | 4836 |
| linear regression | linear | SPD, acceleration | 31.958 | 43.942 | -2.624 | 0.350 | 4836 |
| physics calibration | physics | a*v, v^3, grade, v, AIR | 25.028 | 36.054 | -0.468 | 0.562 | 4836 |
| random forest | rf | SPD, accel_mps2, ACC, BRK, ECO_bin, AIR_bin, grade_pct | 5.425 | 9.433 | -0.466 | 0.970 | 4836 |
