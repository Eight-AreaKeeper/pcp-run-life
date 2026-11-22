# PCP run-life model

A survival model that predicts how long a heavy-oil progressing-cavity pump (PCP) will run after a
downhole repair, and which operating conditions and repair choices go with shorter or longer runs.

- **Input:** daily production, well jobs and failure log (parquet).
- **Output:**
  - each well's probability of failing within 90 producing days, its expected run life and its
    survival curve;
  - hazard ratios for each factor;
  - Kaplan-Meier run life by group.
- **Model:** Cox proportional hazards (semi-parametric survival analysis, handles censored runs),
  robust standard errors clustered by well.

## Results

All numbers are on held-out data: trained on runs that started before 2025-01-01, tested on the
3,402 runs that started in 2025–2026.

| | Cox model | Simple rule (failures in the past year) | Same risk for everyone (Kaplan-Meier) |
|---|---|---|---|
| C-index | **0.644** | 0.604 | 0.500 |
| AUC, failure within 90 producing days | **0.722** | 0.660 | 0.500 |
| Brier score, 90 days (lower is better) | **0.207** | n/a | 0.243 |
| Top-50 flagged wells that failed within 90 days | **38 / 50** | 35 / 50 | n/a |

- Base rate: 43% of each well's first test run fails within 90 days. The model's top-50 list hits 76%.
- At a 50% threshold the model is 67.6% accurate with 79% recall. Predicting "fails" for every run
  gives 58.7%, and the simple rule (3 or more failures in the past year) gives 64.3%.
- Calibration is good: in every risk decile, the predicted and observed failure shares are within
  about 5 points (`outputs/calibration.csv`).

What goes with shorter or longer runs (hazard ratio; above 1 means shorter runs; Cox model on the
training runs; robust standard errors clustered by well):

| factor | hazard ratio (95% CI) | reads as |
|---|---|---|
| thorough repair (Workover or new rod string) vs patch (other rod-truck job) | **0.555** (0.525–0.587) | 44% lower failure risk |
| failures in the past 365 days, per 1 sd (2.5 failures) | 1.23 (1.18–1.27) | wells that fail often keep failing |
| oil rate, per 1 sd (4.4 m³/d) | 1.11 (1.08–1.14) | |
| pump RPM, per 1 sd (119 rpm) | 1.08 (1.05–1.11) | faster pumps fail sooner |
| water cut, per 1 sd (0.16) | 1.07 (1.03–1.10) | |
| previous run length, per 1 sd (117 days) | 0.88 (0.85–0.91) | long runs follow long runs |

Observed median run life (Kaplan-Meier, producing days):
- 50 days after a patch repair vs 123 days after a thorough repair.
- By RPM quartile, lowest to highest: 94, 74, 64 and 58 days.

These are associations, not causes. For example, wells chosen for a thorough repair may differ in
ways the data does not show.

## Method

1. **Data** (`src/build_runs.py`). We keep heavy-oil producers only: 3.15M of the 5.08M daily rows,
   1,863 wells.
   - A day counts as a **downhole day** when `hours_down > 0` and `downtime_code_1` mentions
     "downhole" (rods, pump, tubing, rod string, well torqued, unspecified downhole).
   - Downhole days of one well that are at most 4 days apart are merged into one **failure event**.
     This gives 10,393 events on 1,191 wells.
2. **Check against the official failure log.**
   - 91.6% of the logged failures (2,298 of 2,509) fall within 7 days of one of our events.
   - Only 22.4% of our events since the log started appear in the log, so the log seems to record
     only part of the downhole failures.
   - Results are in `outputs/event_check.csv`.
3. **Runs.**
   - A run starts on the first producing day after an event and ends when the next event starts
     (event = 1). If no failure follows, it ends at the last record (event = 0, censored).
   - Length is counted in producing days, so long shut-ins do not count.
   - This gives 10,209 runs; 90% end in failure and the median is 65 producing days.
4. **Features**, all known on the restart day:
   - RPM, water cut and oil rate over the 30 producing days before the failure;
   - failures in the past 365 days;
   - previous run length;
   - repair type, from well_jobs within 7 days of the event.
   Every cleaning step and its row counts are listed in `outputs/cleaning_log.csv`.
5. **Model** (`src/train_eval.py`).
   - Time split: train before 2025-01-01, test after; never a random split.
   - Feature prep is fitted on the training set only: clip to the 1st–99th percentile, fill with
     the median plus a missing flag, and standardize.
   - Cox proportional hazards model (lifelines), L2 penalty 0.01, robust standard errors clustered
     by well.
   - We compare it with the simple rule and with Kaplan-Meier.
6. **Use.** `predict_fail_prob()` in `src/runlife_prep.py` gives each running well's probability of
   failing within 90 producing days. Probability × (repair cost + lost oil value) ranks the
   weekly watchlist. `src/example_predict.py` shows all outputs on made-up wells.

**Limitations.**
- Failure events come from downtime codes, not from a confirmed failure list.
- Costs exist only as field estimates (`field_est_total`).
- Repair type is inferred from job records near the event, and 11% of events have no matching job
  (`unknown`).
- The Cox model assumes each factor scales risk by the same amount at every point of a run
  (proportional hazards) and that each factor acts linearly on log-risk.

## Reproduce

```
pip install -r requirements.txt
# put the three source files in data/ (not included in this repo):
#   data/daily_production.parquet  data/well_jobs.parquet  data/production_failures.parquet
python src/build_runs.py --data data --out outputs                       # ~30 s
python src/train_eval.py --runs outputs/wells/runs.parquet --out outputs  # ~1 min
python src/example_predict.py                                            # example outputs
```

Without the data, `python src/make_fake_runs.py --out outputs/wells/runs.parquet` makes a simulated
file for testing the model code. Its numbers are fake.

## Repository layout

| path | contents |
|---|---|
| `src/build_runs.py` | raw data → failure events → runs.parquet (cleaning, events, features) |
| `src/runlife_prep.py` | runs.parquet contract, feature prep, `predict_fail_prob` |
| `src/train_eval.py` | Kaplan-Meier, Cox model, evaluation against baselines |
| `src/example_predict.py` | example outputs: one well, hazard ratios, group run life |
| `src/make_fake_runs.py` | simulated runs.parquet for testing |
| `outputs/*.csv`, `outputs/model_meta.json` | aggregate results (no well IDs) |
| `outputs/wells/` | well-level files and the fitted model; gitignored, never public |
| `data/` | raw source data; gitignored |

Output files: `km_curves.csv` (survival by group, first 365 days), `cox_results.csv` (hazard
ratios), `model_eval.csv` (test metrics), `calibration.csv` (by decile), `cleaning_log.csv`,
`event_check.csv`, `model_meta.json`.

### runs.parquet contract

One row per run. Required columns:
- `run_id`, `uwi`
- `start_date` (the restart day)
- `duration_days` (producing days, > 0)
- `event` (1 = failed, 0 = censored)

Optional features: `rpm`, `water_cut`, `oil_rate`, `n_fail_365`, `prev_run_days`, `repair_type`
(`patch` / `thorough` / `other` / `unknown`, with `patch` as reference). To add a feature, add a
numeric column and list it in `NUMERIC_FEATURES` in `src/runlife_prep.py`.

## External resources

- [lifelines](https://lifelines.readthedocs.io/) (Cox model, Kaplan-Meier), scikit-learn (AUC),
  DuckDB, pandas.
- Cox, D. R. (1972). Regression models and life-tables. *JRSS B*.
