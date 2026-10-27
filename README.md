# Estimating EV Battery Current from Driving Signals

**Python for Engineers — Term Project (From Data to Insight)**
Türk-Alman Üniversitesi, Winter 2026 · Dr. Berfin Aktaş · Individual project
Can Keçilioğlu

[Repository](https://github.com/cankecilioglu/ev-telemetry-project) ·
[Case study](https://cankecilioglu.dev/work/ev-telemetry)

A modular Python analysis of a real electric-vehicle telemetry dataset: clean the
data, describe how the car is actually driven, and estimate the **signed battery
current** from simultaneous driving signals with a physics-based calibration, a
Random Forest and a mean baseline.

---

## 1. Research question

> Can the battery current demand and regenerative-braking recovery of an electric
> vehicle (`CUR`, in A) be estimated from simultaneous vehicle kinematics
> (`SPD`, `AX`), pedal inputs (`ACC`, `BRK`) and auxiliary states (`ECO`, `AIR`)?

The target is the **signed** current: positive = power drawn from the pack,
negative = energy flowing back during regenerative braking. A single signed
target is used because it is one physical quantity measured by one sensor;
regenerative operation is then analysed separately by braking regime in the
error analysis, instead of being modelled by a second, silently different
target.

Only information available **at the same instant** is used — the project
estimates, it does not forecast.

## 2. Dataset, licence and citation

| Item | Value |
| --- | --- |
| Dataset | Electric Vehicle Telemetry Dataset (JAC iEV40, 9 drivers, same route) |
| Source | [Kaggle — marceloreis76/electric-vehicle-telemetry-dataset](https://www.kaggle.com/datasets/marceloreis76/electric-vehicle-telemetry-dataset) |
| Licence (dataset) | **GPL-3.0**, as declared on the Kaggle dataset page (verified via the Kaggle API) |
| Reference publication | dos-Reis, M., Frison, C. I., Teixeira, F. C., & Marques-Neto, H. T. (2024). *Using Electric Vehicle Driver's Driving Mode for Trip Planning and Routing.* Journal of Internet Services and Applications, 15(1), 410–423. [doi:10.5753/jisa.2024.3805](https://doi.org/10.5753/jisa.2024.3805) |
| Collection | October 2023, Poços de Caldas (MG) → São João da Boa Vista (SP) and back, Brazil |
| Size | 24,277 rows × 27 columns, ~76 km per trip, ~1–2 s sampling |

The raw CSV is **not redistributed** in this repository (GPL-3.0 dataset, and it
keeps the clone small); `data/raw/` is ignored by git. Downloading the single
`dataset.csv` and placing it there is the only manual step.

## 3. Quick start

```bash
# 1. environment
python -m venv .venv
.venv\Scripts\activate          # Windows  (source .venv/bin/activate on Linux/macOS)
pip install -r requirements.txt

# 2. data
#    download dataset.csv from the Kaggle link above into data/raw/

# 3. run
python main.py --data data/raw/dataset.csv

# 4. tests
pytest -q

# 5. optional: the narrative notebook (needs Jupyter)
pip install -r requirements-notebook.txt
jupyter lab exploration.ipynb
```

`exploration.ipynb` is committed **with its outputs**, so it can be read on
GitHub without installing anything; running it reproduces every number in this
README from the raw CSV.

Stages can be run separately:

```bash
python main.py --data data/raw/dataset.csv --stage clean      # cleaning + quality report
python main.py --data data/raw/dataset.csv --stage explore    # + figures
python main.py --data data/raw/dataset.csv --stage baseline   # + split + baseline MAE
```

Every figure and table is written to `results/`; nothing is shown on screen, so
the pipeline also runs on a machine without a display.

## 4. Repository layout

```
ev-telemetry-project/
├── main.py                  # CLI entry point
├── exploration.ipynb        # narrative notebook (executed outputs included)
├── src/
│   ├── config.py            # paths, physical constants, validation ranges
│   ├── io.py                # file loading with explicit error messages
│   ├── cleaning.py          # timestamp merging, unit conversion, filters
│   ├── quality.py           # data-quality tables
│   ├── features.py          # derived variables (acceleration, grade, power)
│   ├── exploration.py       # EDA figures
│   ├── modeling.py          # split, metrics, baseline, physics, RF, errors
│   ├── diagnostics.py       # model-comparison and residual figures
│   ├── plotting.py          # figure style and saving
│   └── reporting.py         # markdown/CSV table export
├── tests/
│   ├── test_cleaning.py
│   ├── test_features.py
│   ├── test_modeling.py
│   └── test_exploration.py
├── data/raw/                # dataset.csv (not committed)
├── results/
│   ├── figures/             # fig01 … fig07 (.png)
│   └── tables/              # .csv + .md reports
├── requirements.txt
├── requirements-notebook.txt
└── conftest.py
```

## 5. Data-quality report

Generated automatically by `src/quality.py` → [`results/tables/data_quality_report.md`](results/tables/data_quality_report.md).

| Issue | Count | Decision |
| --- | --- | --- |
| Raw rows read | 24,277 | — |
| Unparseable UTC timestamp (month 49, hour 83, …) | 69 | Row dropped: the observation cannot be placed in time. |
| Timestamp outside 2023 | 2 | Row dropped (two-digit year 0 → 2000). |
| Timestamp contradicting the recording order | 26 | `id` is 100 % monotonic with the odometer, so it defines the true sequence. Rows stamped 16–20 h away from their neighbours (corrupted hour/minute) were dropped. |
| Duplicate (trip, timestamp) | 0 | After the previous step no duplicates remained. |
| Physically impossible values (`SPD`, `ACC`, `BRK`, `CH`, `CUR` ranges) | 0 | No row had to be dropped for this reason; the checks stay in place and are unit-tested. |
| Missing cells in the raw file | 0 | Missing data is **encoded with sentinels**, not with `NaN` (see the next three rows). |
| `ALT == 0` (missing altitude) | 4 | Flagged `alt_missing`; the route is at 850–1350 m, so 0 is not a measurement. |
| `VOL` outside 250–420 V | 18,049 (75 %) | The pack-voltage channel is unusable on 7 of 9 trips; flagged `vol_valid`, rows kept. |
| GPS outside the Brazil bounding box | 21 | Flagged `gps_valid`; GPS is not a model input. |
| **Rows remaining** | **24,180** | **0.40 % removed** |

Findings worth highlighting in the report:

- **The timestamp columns are mostly good but not always.** 97 rows carry
  impossible or contradictory times. `pandas.to_datetime` silently *rolls over*
  out-of-range values (hour 83 becomes three days later, minute 95 becomes
  13:35), so the components are validated explicitly in `build_timestamp` —
  without that check, three corrupted rows would have been moved days into the
  future without any warning.
- **The file is not stored in time order.** Rows from the nine trips are
  interleaved; the pipeline sorts per trip before anything else.
- **`VOL` is broken on trips 1–6 and 9** (values 0–25 V instead of ~350 V),
  while trips 7–8 carry a plausible 299–381 V. See
  [`fig02_voltage_channel_quality.png`](results/figures/fig02_voltage_channel_quality.png).
  Because power `P = V·I` needs a trustworthy `V`, the model predicts **current**
  and uses the nominal pack voltage (350 V) in the physics term, with the
  sensitivity to that assumption checked explicitly.
- **Transmission loss is visible in trip 9** (one 75-minute hole, 88 km on the
  odometer but only 792 samples). The published description also mentions trip 6,
  but no gap of that kind remains in this file — the report states what the data
  show, not what the article says.
- **Sampling is ~1–2 s** (median 2 s), not the "high frequency" a reader might
  expect; this limits how well pedal transients can be resolved.
- **The accelerometer is not usable as the acceleration source.** `AX` has a
  per-trip offset of 0.07–3.13 m/s² and correlates with the differentiated speed
  at only r = 0.14; it correlates with current at r = 0.22, while `dSPD/dt` reaches
  r = 0.50. The pipeline therefore differentiates the speed channel and documents
  this as a sensor-quality finding instead of feeding a biased axis into the model.

## 6. Sign convention of the target

| Check | Value | Interpretation |
| --- | --- | --- |
| `CUR < 0` | 6,596 rows (27.3 %) | Regenerative / recuperation operation |
| … while braking (`BRK > 0`) | 1,486 | Classic lift-off / brake regen |
| … while coasting (no pedal) | 4,735 | Overrun recuperation |
| … while accelerating (`ACC > 0`) | 375 | Contradictory: sensor noise or torque blending |
| `CUR > 0` while braking | 1,375 | Hydraulic braking before regen engages |
| Range | −52 … +301 A | Plausible for a ~45 kW pack on a compact EV |

So the dataset uses **positive = discharge, negative = charge**, which matches
the intended definition. The contradictory cases are kept (they are real sensor
output) and discussed in the error analysis rather than deleted.

## 7. Physics-based calibration

The reference model is a longitudinal power balance evaluated at the pack:

```
I_batt = [ m·a·v + ½·ρ·C_d·A·v³ + m·g·sin(θ)·v + C_rr·m·g·cos(θ)·v ] / (η · V_nom)  +  P_aux/V_nom
```

| Symbol | Meaning | Source |
| --- | --- | --- |
| `v`, `a` | speed, longitudinal acceleration | `SPD/3.6`; `a = dSPD/dt` per trip (not `AX`, see §5) |
| `θ` | road grade | `arctan(ΔALT/Δdistance)` over a trailing 15-sample (~30 s) window, clipped to ±15 % |
| `m`, `C_d`, `A`, `C_rr`, `ρ` | vehicle and environment parameters | JAC iEV40 public data (priors only) |
| `η` | drivetrain efficiency | 0.90 assumption; the fitted coefficients absorb the true value |
| `V_nom` | nominal pack voltage, 350 V | because `VOL` is corrupt on 7 trips |
| `P_aux` | base + air-conditioning load | calibrated from `AIR` |

With `cos θ ≈ 1` (below 1 % error at the 15 % clip) the equation is linear in six
physically meaningful coefficients (`m/η`, `½ρC_dA/η`, `m·g/η`, `C_rr·m·g/η`,
`P_base`, `P_air`). It is fitted by ordinary least squares **on the training set
only**, and the fitted quantities are translated back into apparent mass, `C_dA`,
`Crr` and auxiliary power for a physical sanity check.

Assumptions: quasi-steady longitudinal motion, no wheel slip, one constant
efficiency for traction and regeneration, auxiliary load independent of speed,
and grade estimated from 1 m-resolution altitude data. The grade is a trailing
estimate, so it uses only information available at the sampling instant.

## 8. Validation protocol

- **Time-ordered split**: rows sorted by `timestamp`; the first 80 % are training,
  the last 20 % test. No shuffling anywhere. The cut falls on 26 October, so the
  test set contains **unseen drivers and a new day**.
- **No leakage**: every learned quantity (coefficients, imputer, hyper-parameters)
  is fitted on the training slice only; `TimeSeriesSplit(5)` is used inside the
  training set for tuning.
- **Four methods on the same test rows**: training-mean baseline, optional linear
  regression on speed and acceleration, physics calibration and Random Forest —
  compared by **MAE in amperes**.
- **Per-driver robustness check**: a second split holds out the last 20 % of
  *each* trip, so all nine drivers appear in the test set.
- **Error analysis**: residuals against speed, grade and operating regime, plus
  a bias discussion that separates evidence from hypotheses.

## 9. Current results

Training: 19,344 rows (trips 1–6, the first part of 7, trip 9).
Test: 4,836 rows (end of trip 7 and all of trip 8).

| Model | Features | Test MAE [A] | Test RMSE [A] | Bias [A] | R² |
| --- | --- | --- | --- | --- | --- |
| Baseline (training mean) | — | **40.35** | 54.58 | −2.84 | −0.003 |
| Linear regression (reference) | `SPD`, `a` | 31.96 | 43.94 | −2.62 | 0.350 |
| Physics calibration | `a·v`, `v³`, grade, `v`, `AIR` | 25.03 | 36.05 | −0.47 | 0.562 |
| Random Forest | `SPD`, `a`, `ACC`, `BRK`, `ECO`, `AIR`, grade | **5.43** | 9.43 | −0.47 | 0.970 |

Random Forest: cross-validated MAE inside training = 6.96 A; selected
`n_estimators = 200`, `max_depth = 12`, `min_samples_leaf = 3`. Feature
importance is dominated by the accelerator pedal (0.90), then speed (0.07) and
acceleration (0.015); `ECO`, grade, `AIR` and `BRK` together contribute < 2 %.

Why is the forest so much better? The accelerator pedal alone correlates with
current at r = +0.91 (fig03), so the forest mostly learns the pedal-to-current
map of this vehicle, plus corrections for speed and grade. That is a legitimate
*simultaneous* signal, but it is also a lesson: the physics model is the one that
transfers to another vehicle, while the forest is tied to this pedal calibration.

Physics coefficients translated back to vehicle parameters:

| Coefficient | Fitted | Prior / implied | Comment |
| --- | --- | --- | --- |
| `m/η` | 970 kg | prior 1,706 kg → **implied 873 kg** | 43 % below the 1,535 kg kerb mass |
| `½ρC_dA/η` | 0.612 | prior 0.461 → **`C_dA` = 0.917 m²** | prior `C_dA` = 0.69 m² |
| `m·g/η` | 10,453 N | **implied 959 kg** | agrees with the traction mass within 10 % |
| `C_rr·m·g/η` | 90.4 N | **`C_rr` = 0.0095** | prior 0.012 |
| `P_base` | 3,067 W | — | base electrical load |
| `P_air` | 1,148 W | — | air-conditioning load |

The two independent mass estimates (traction and grade) agree within 10 %, which
is the main evidence that the fitted equation is physically coherent. The absolute
mass is 43 % low; the most plausible explanation is that `dSPD/dt` over 2 s
under-represents short acceleration peaks and that a single efficiency cannot
describe traction and regeneration at once. This is a discussion point, not a
hidden correction — the coefficient table is written to
[`physics_coefficients.md`](results/tables/physics_coefficients.md) unchanged.

**Per-driver robustness check** (last 20 % of every trip held out, n = 4,835):

| Model | MAE [A] | Bias [A] |
| --- | --- | --- |
| Baseline | 38.48 | −2.66 |
| Linear | 33.20 | −1.57 |
| Physics | 27.24 | +0.54 |
| Random Forest | 6.21 | −0.68 |

The ranking is unchanged, so the result is not an artefact of which trips happen
to fall after the global cut.

**Error analysis** (`fig06`, `error_by_*.md`):

- The baseline has a systematic **−35.6 A bias above 80 km/h** and −28.7 A uphill:
  the mean is dominated by low-demand samples and cannot follow demand.
- Physics error grows from 9.5 A (0–20 km/h) to 38.9 A (80–100 km/h); at 100+ km/h
  it is 45 A. Aerodynamic and rolling terms are calibrated on average conditions,
  and the single efficiency cannot follow the traction/regen asymmetry.
- The forest stays between 2.6 and 7.4 A in every speed band, and between 3.9 A
  (braking) and 6.5 A (coasting) per regime. Its largest remaining bias is
  −3.5 A at 80–100 km/h.
- Braking is the easiest regime for the physics model (14.6 A), because the
  recovery term is proportional to speed and dominates the balance.

**ECO mode — a confounded natural experiment.** Trip 4 switches ECO four times,
so it looks like a clean within-driver test. It is not: ECO-on windows also have
twice the accelerator pedal (27 % vs 12 %) and sit on slightly uphill sections
(+2.2 % vs −1.2 % grade), so raw current is *higher* with ECO on (49.5 A vs
19.7 A). Comparing only samples in the same speed band **and** the same
accelerator-pedal quintile reverses the sign: ECO on is 16–34 A lower, with a
matched average of −20 A (regression-adjusted estimate: −15 A). Both the raw and
the matched view are shown in
[`fig04_eco_regime.png`](results/figures/fig04_eco_regime.png), because the
contrast is the finding.

## 10. How the proposal feedback is addressed

1. **Signed current vs separate regeneration** — §6: one signed target, sign
   convention verified against braking and coasting counts; regen is analysed
   per regime, not as a hidden second target.
2. **Physics formula, variables, assumptions** — §7, fitted by OLS on training
   rows only, with coefficients translated back to vehicle parameters.
3. **Implausible speeds, accelerations, voltages, currents, timestamps** — §5:
   explicit range checks, timestamp validation against the recording order,
   sentinel flags, and the documented `AX` offset finding.
4. **Licence and citation** — §2 (GPL-3.0 confirmed from the Kaggle metadata;
   citation of the JISA article).
5. **Model comparison** — §8–9: mean baseline + physics calibration + Random
   Forest, plus the optional speed/acceleration linear regression as an extra
   reference.

## 11. Status and roadmap

- [x] Cleaning pipeline, unit conversion, quality report, automated tests
- [x] Exploratory figures: time series, voltage quality, pedal response, ECO
- [x] Chronological split and baseline MAE
- [x] Physics calibration + Random Forest + comparison table
- [x] Error analysis by speed / grade / regime / driver
- [x] Narrative notebook with executed outputs (`exploration.ipynb`)
- [ ] Final README pass, ZIP package, slides, portfolio page

## 12. Reproducibility

- All random seeds fixed (`RANDOM_SEED = 42` in `src/config.py`).
- Paths are resolved from the repository root — the project runs after cloning.
- `python main.py --data <path>` is the only entry point; `--stage` runs one part,
  `--no-tune` skips the grid search for a fast smoke run.
- Test suite: `pytest -q` → **28 tests** (11 cleaning, 4 feature engineering,
  11 split/modelling, 2 exploration), all passing.
- `exploration.ipynb` was executed top-to-bottom with the same environment
  (`jupyter nbconvert --execute`); its outputs are part of the committed file.
- The full pipeline takes ~1 minute on a laptop, dominated by the 18-point
  Random Forest grid search.

## 13. Licence

- **Code**: [MIT](LICENSE) © 2026 Can Keçilioğlu.
- **Dataset**: GPL-3.0, © the original dataset authors (see §2). The raw CSV is
  not redistributed here; only the derived figures, tables and code are.
