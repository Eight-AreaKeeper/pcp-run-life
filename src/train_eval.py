"""Step 6: train and evaluate the run-life (survival) model.

Input : runs.parquet, one row per run (see README.md for the column contract).
Output: aggregate CSVs in outputs/ (safe to publish) and the fitted model in
        outputs/wells/ (gitignored, used by score_wells.py in step 8).

    python src/train_eval.py --runs outputs/wells/runs.parquet --out outputs

What it does
1. Checks that runs.parquet follows the contract.
2. Splits by time: runs that start before --split-date train, later runs test.
3. Kaplan-Meier curves overall, by repair type and by RPM quartile.
4. Cox proportional hazards model on the training runs, with robust standard
   errors clustered by well.
5. Scores the test runs and compares the model with two baselines:
   Kaplan-Meier (everyone gets the same risk) and the simple rule
   (more failures in the past year = more risk).
6. Metrics: C-index, 90-day AUC, calibration by decile, top-50 hits.
"""
import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runlife_prep import (CATEGORICAL_FEATURES, HORIZON, NUMERIC_FEATURES,  # noqa: E402
                          SIMPLE_RULE_COL, TOP_K, Prep, check_contract)


# ---------------------------------------------------------------- helpers
def km_table(df: pd.DataFrame, group_col: str | None, label: str) -> pd.DataFrame:
    parts = []
    groups = [("all", df)] if group_col is None else df.groupby(group_col, observed=True)
    for g, d in groups:
        if len(d) < 30:
            continue
        km = KaplanMeierFitter().fit(d["duration_days"], d["event"])
        sf = km.survival_function_.reset_index()
        sf.columns = ["day", "survival"]
        sf = sf[sf["day"] <= 365]
        sf.insert(0, "group", str(g))
        sf.insert(0, "split_by", label)
        sf["n_runs"] = len(d)
        sf["median_run_days"] = km.median_survival_time_
        parts.append(sf)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def horizon_label(df: pd.DataFrame, h: int = HORIZON) -> pd.Series:
    """1 = failed within h days, 0 = ran past h days, NaN = censored before h (unknown)."""
    y = pd.Series(np.nan, index=df.index)
    y[(df["event"] == 1) & (df["duration_days"] <= h)] = 1
    y[df["duration_days"] > h] = 0
    return y


def top_k_hits(test: pd.DataFrame, score: pd.Series, y: pd.Series, k: int = TOP_K) -> dict:
    """Flag k wells: each well's first test run, ranked by score. Count failures within the horizon."""
    first = test.sort_values("start_date").groupby("uwi").head(1).index
    s = score.loc[first]
    lab = y.loc[first]
    known = lab.notna()
    ranked = s[known].sort_values(ascending=False, kind="mergesort")
    top = ranked.index[:k]
    return dict(flagged=int(len(top)), hits=int(lab.loc[top].sum()),
                base_rate=float(lab[known].mean()))


def calibration(pred: pd.Series, y: pd.Series, bins: int = 10) -> pd.DataFrame:
    d = pd.DataFrame({"pred": pred, "y": y}).dropna()
    d["bin"] = pd.qcut(d["pred"].rank(method="first"), bins, labels=False) + 1
    return (d.groupby("bin")
             .agg(n_runs=("y", "size"), mean_predicted=("pred", "mean"), observed=("y", "mean"))
             .reset_index())


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default="outputs/wells/runs.parquet")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--split-date", default="2025-01-01")
    ap.add_argument("--penalizer", type=float, default=0.01,
                    help="small L2 penalty keeps the Cox fit stable")
    a = ap.parse_args()

    out = Path(a.out)
    (out / "wells").mkdir(parents=True, exist_ok=True)

    runs = pd.read_parquet(a.runs)
    feats = check_contract(runs)
    split = pd.Timestamp(a.split_date)
    train = runs[runs["start_date"] < split].copy()
    test = runs[runs["start_date"] >= split].copy()
    print(f"runs: {len(runs):,}  train (< {split.date()}): {len(train):,}  test: {len(test):,}")
    if len(train) < 200 or len(test) < 50:
        raise SystemExit("too few runs on one side of the split; check start_date or --split-date")

    # 1. Kaplan-Meier curves (all runs; descriptive, not used for scoring)
    km_parts = [km_table(runs, None, "all")]
    if "repair_type" in runs:
        km_parts.append(km_table(runs.assign(repair_type=runs["repair_type"].fillna("unknown")),
                                 "repair_type", "repair_type"))
    if "rpm" in runs:
        q = pd.qcut(runs["rpm"], 4, labels=["Q1 lowest", "Q2", "Q3", "Q4 highest"])
        km_parts.append(km_table(runs.assign(rpm_quartile=q), "rpm_quartile", "rpm_quartile"))
    pd.concat(km_parts, ignore_index=True).to_csv(out / "km_curves.csv", index=False)

    # 2. Cox model on training runs
    prep = Prep().fit(train, feats)
    Xtr = prep.transform(train)
    fit_df = Xtr.assign(duration_days=train["duration_days"].values,
                        event=train["event"].values, uwi=train["uwi"].values)
    cph = CoxPHFitter(penalizer=a.penalizer)
    cph.fit(fit_df, duration_col="duration_days", event_col="event",
            cluster_col="uwi", robust=True)

    s = cph.summary
    cox = pd.DataFrame({
        "covariate": s.index,
        "coef": s["coef"].values,
        "hazard_ratio": s["exp(coef)"].values,
        "hr_ci_low": s["exp(coef) lower 95%"].values,
        "hr_ci_high": s["exp(coef) upper 95%"].values,
        "p_value": s["p"].values,
    })
    def unit(c):
        base = c.split("=")[0]
        if c.endswith("_missing"):
            return "missing vs present"
        if "=" in c:
            return f"vs {CATEGORICAL_FEATURES[base]}"
        st = prep.stats[c]
        return f"per 1 sd ({st['raw_sd']:.3g} raw units)"
    cox["reads_as"] = cox["covariate"].map(unit)
    cox["significant_5pct"] = cox["p_value"] < 0.05
    cox.sort_values("hazard_ratio", ascending=False).to_csv(out / "cox_results.csv", index=False)

    # 3. Score the test runs
    Xte = prep.transform(test)
    risk_cox = pd.Series(cph.predict_partial_hazard(Xte).values.ravel(), index=test.index)
    surv_h = cph.predict_survival_function(Xte, times=[HORIZON]).T.iloc[:, 0]
    p_fail_cox = pd.Series(1 - surv_h.values, index=test.index)

    km_train = KaplanMeierFitter().fit(train["duration_days"], train["event"])
    p_fail_km = pd.Series(1 - float(km_train.predict(HORIZON)), index=test.index)

    if SIMPLE_RULE_COL in test:
        risk_rule = test[SIMPLE_RULE_COL].fillna(0).astype(float)
    else:
        risk_rule = pd.Series(0.0, index=test.index)

    # 4. Metrics
    y90 = horizon_label(test)
    known = y90.notna()
    def cidx(risk):
        return concordance_index(test["duration_days"], -risk, test["event"])
    def auc(score):
        yy = y90[known]
        return roc_auc_score(yy, score[known]) if yy.nunique() == 2 else np.nan

    rows = []
    for name, risk, p in [("cox_model", risk_cox, p_fail_cox),
                          ("simple_rule_past_year_failures", risk_rule, None),
                          ("kaplan_meier_same_for_all", pd.Series(0.0, index=test.index), p_fail_km)]:
        is_km = name == "kaplan_meier_same_for_all"
        # Kaplan-Meier gives every well the same risk, so it cannot rank wells: top-50 is n/a
        hits = top_k_hits(test, risk, y90)
        if is_km:
            hits.update(flagged=np.nan, hits=np.nan)
        rows.append(dict(
            model=name,
            c_index=round(cidx(risk), 4),
            auc_90d=0.5 if is_km else round(auc(risk), 4),
            brier_90d=round(float(((p[known] - y90[known]) ** 2).mean()), 4) if p is not None else np.nan,
            top50_flagged=hits["flagged"],
            top50_failed_within_90d=hits["hits"],
            first_run_fail_rate_90d=round(hits["base_rate"], 4),
            test_runs=len(test),
            test_runs_with_known_90d_outcome=int(known.sum()),
        ))
    ev = pd.DataFrame(rows)
    ev.to_csv(out / "model_eval.csv", index=False)
    calibration(p_fail_cox, y90).to_csv(out / "calibration.csv", index=False)

    # 5. Save the model for step 8 (well-level, so it stays out of the public repo)
    with open(out / "wells" / "cox_model.pkl", "wb") as f:
        pickle.dump(dict(model=cph, prep=prep, horizon=HORIZON, split_date=str(split.date()),
                         features=feats), f)
    meta = dict(split_date=str(split.date()), horizon_days=HORIZON, features=feats,
                train_runs=len(train), test_runs=len(test),
                train_failures=int(train["event"].sum()), test_failures=int(test["event"].sum()))
    (out / "model_meta.json").write_text(json.dumps(meta, indent=2))

    # 6. Short printout
    print("\nHazard ratios (>1 = shorter runs), training data:")
    print(cox[["covariate", "hazard_ratio", "hr_ci_low", "hr_ci_high", "p_value", "reads_as"]]
          .round(3).to_string(index=False))
    print("\nTest-set evaluation:")
    print(ev.to_string(index=False))
    print(f"\nWrote km_curves.csv, cox_results.csv, model_eval.csv, calibration.csv, "
          f"model_meta.json to {out}/ and the model to {out}/wells/cox_model.pkl")


if __name__ == "__main__":
    main()
