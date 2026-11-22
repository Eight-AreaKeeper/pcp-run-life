"""Examples of the three things the run-life model can output.

    python src/example_predict.py        (run from the run-life folder, after train_eval.py)

1. One well:  its whole survival curve, expected run life and 90-day failure probability
2. All wells: hazard ratios (which conditions and repairs shorten or extend run life)
3. Groups:    Kaplan-Meier median run life by repair type and by RPM quartile
The wells in part 1 are made up; change the numbers to try your own.
"""
import pickle
import sys
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runlife_prep import predict_fail_prob  # noqa: E402

OUT = Path("outputs")
bundle = pickle.load(open(OUT / "wells" / "cox_model.pkl", "rb"))
model, prep = bundle["model"], bundle["prep"]
pd.set_option("display.width", 200)

# ------------------------------------------------------------ 1. one well
print("=" * 70)
print("1. ONE WELL: survival curve, expected run life, 90-day failure probability")
print("=" * 70)
wells = pd.DataFrame([
    dict(well="A calm",              rpm=150, water_cut=0.60, oil_rate=3.0, n_fail_365=0, prev_run_days=400, repair_type="thorough"),
    dict(well="B typical",           rpm=250, water_cut=0.80, oil_rate=5.0, n_fail_365=2, prev_run_days=90,  repair_type="patch"),
    dict(well="C high risk",         rpm=400, water_cut=0.95, oil_rate=8.0, n_fail_365=6, prev_run_days=20,  repair_type="patch"),
    dict(well="D = C but thorough",  rpm=400, water_cut=0.95, oil_rate=8.0, n_fail_365=6, prev_run_days=20,  repair_type="thorough"),
])
print("\nInput (all known on the restart day):")
print(wells.to_string(index=False))

X = prep.transform(wells)
days = [30, 60, 90, 180, 365]
surv = model.predict_survival_function(X, times=days).T
surv.columns = [f"still_running_day_{d}" for d in days]
res = pd.DataFrame({"well": wells["well"]})
res["p_fail_90d"] = predict_fail_prob(bundle, wells).round(3).values
res["expected_run_days_median"] = model.predict_median(X).values
res = pd.concat([res, surv.reset_index(drop=True).round(3)], axis=1)
print("\nOutput:")
print(res.to_string(index=False))

# ------------------------------------------------------------ 2. hazard ratios
print("\n" + "=" * 70)
print("2. ALL WELLS: hazard ratios (>1 = shorter runs, <1 = longer runs)")
print("=" * 70)
cox = pd.read_csv(OUT / "cox_results.csv")
cox["effect"] = ((cox["hazard_ratio"] - 1) * 100).round(0).astype(int).map(lambda v: f"{v:+d}% risk")
print(cox[["covariate", "hazard_ratio", "hr_ci_low", "hr_ci_high", "p_value", "reads_as", "effect"]]
      .round(3).to_string(index=False))

# ------------------------------------------------------------ 3. KM groups
print("\n" + "=" * 70)
print("3. GROUPS: observed median run life (Kaplan-Meier, producing days)")
print("=" * 70)
km = pd.read_csv(OUT / "km_curves.csv")
summary = km.groupby(["split_by", "group"], sort=False)[["n_runs", "median_run_days"]].first().reset_index()
for d in [90, 180]:
    s = (km[km["day"] <= d].groupby(["split_by", "group"], sort=False)["survival"].last()
           .round(3).rename(f"still_running_day_{d}"))
    summary = summary.merge(s.reset_index(), on=["split_by", "group"])
print(summary.to_string(index=False))
