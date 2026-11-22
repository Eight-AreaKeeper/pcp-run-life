"""Shared pieces for the run-life model: the runs.parquet contract and the feature prep.

Kept in its own module so the saved model (outputs/wells/cox_model.pkl) can be
loaded from any script, for example score_wells.py in step 8:

    import pickle, sys; sys.path.insert(0, "src")
    bundle = pickle.load(open("outputs/wells/cox_model.pkl", "rb"))
    p90 = predict_fail_prob(bundle, runs_df)
"""
import pandas as pd

# ---------------------------------------------------------------- contract
REQUIRED = ["run_id", "uwi", "start_date", "duration_days", "event"]
NUMERIC_FEATURES = ["rpm", "water_cut", "oil_rate", "n_fail_365", "prev_run_days"]
CATEGORICAL_FEATURES = {"repair_type": "patch"}   # column -> reference level
SIMPLE_RULE_COL = "n_fail_365"
HORIZON = 90          # producing days
TOP_K = 50


def check_contract(df: pd.DataFrame) -> list:
    """Stop with a clear message if runs.parquet does not follow the contract."""
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise SystemExit(f"runs.parquet is missing required columns: {missing}")
    if not pd.api.types.is_datetime64_any_dtype(df["start_date"]):
        raise SystemExit("start_date must be a date/datetime column")
    if (df["duration_days"] <= 0).any():
        raise SystemExit("duration_days must be > 0 for every run")
    if not set(df["event"].unique()) <= {0, 1}:
        raise SystemExit("event must be 0 (still running / censored) or 1 (failed)")
    if df["run_id"].duplicated().any():
        raise SystemExit("run_id must be unique")
    feats = [c for c in NUMERIC_FEATURES + list(CATEGORICAL_FEATURES) if c in df.columns]
    absent = [c for c in NUMERIC_FEATURES + list(CATEGORICAL_FEATURES) if c not in df.columns]
    if absent:
        print(f"note: optional feature columns not found, skipped: {absent}")
    if not feats:
        raise SystemExit("no feature columns found; see README.md for the expected names")
    return feats


# ---------------------------------------------------------------- features
class Prep:
    """Learns cleaning rules on the training runs only, then applies them anywhere.

    Numeric: clip to the training 1st-99th percentile, fill gaps with the
    training median, add a 0/1 'was missing' flag, scale to mean 0 / sd 1 so
    hazard ratios read as 'per 1 standard deviation'.
    Categorical: one-hot against the reference level.
    """

    def fit(self, df: pd.DataFrame, feats: list):
        self.num = [c for c in feats if c in NUMERIC_FEATURES]
        self.cat = [c for c in feats if c in CATEGORICAL_FEATURES]
        self.stats = {}
        for c in self.num:
            x = df[c].astype(float)
            lo, hi = x.quantile([0.01, 0.99])
            xc = x.clip(lo, hi)
            med = xc.median()
            filled = xc.fillna(med)
            sd = filled.std() or 1.0
            self.stats[c] = dict(lo=lo, hi=hi, med=med, mean=filled.mean(), sd=sd,
                                 raw_sd=float(sd), add_flag=bool(x.isna().mean() > 0.01))
        self.levels = {c: sorted(df[c].fillna("unknown").astype(str).unique()) for c in self.cat}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)
        for c in self.num:
            s = self.stats[c]
            x = df[c].astype(float)
            if s["add_flag"]:
                out[f"{c}_missing"] = x.isna().astype(int)
            x = x.clip(s["lo"], s["hi"]).fillna(s["med"])
            out[c] = (x - s["mean"]) / s["sd"]
        for c in self.cat:
            ref = CATEGORICAL_FEATURES[c]
            x = df[c].fillna("unknown").astype(str)
            for lvl in self.levels[c]:
                if lvl != ref:
                    out[f"{c}={lvl}"] = (x == lvl).astype(int)
        return out



def predict_fail_prob(bundle: dict, df: pd.DataFrame, horizon: int | None = None) -> pd.Series:
    """Probability that each run fails within `horizon` producing days (default: the trained horizon)."""
    h = horizon or bundle["horizon"]
    X = bundle["prep"].transform(df)
    surv = bundle["model"].predict_survival_function(X, times=[h]).T.iloc[:, 0]
    return pd.Series(1 - surv.values, index=df.index, name=f"p_fail_{h}d")
