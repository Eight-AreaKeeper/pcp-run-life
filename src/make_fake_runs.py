"""Make a FAKE runs.parquet that follows the contract in README.md.

Use it to build and test train_eval.py before the real runs.parquet exists.
The numbers are simulated and mean nothing. Never put them on a slide.

    python src/make_fake_runs.py --out outputs/wells/runs.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DATA_START = pd.Timestamp("2021-01-01")
DATA_END = pd.Timestamp("2026-10-27")
REPAIR_TYPES = ["patch", "thorough", "other", "unknown"]


def simulate(n_wells: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for w in range(n_wells):
        uwi = f"FAKE-{w:05d}"
        facility = f"FAC-{rng.integers(0, 12):02d}"
        # well-level traits that stay roughly fixed
        base_rpm = rng.normal(220, 60)
        base_wc = rng.beta(5, 2)            # water cut 0..1
        base_oil = rng.gamma(2.0, 2.0)      # m3/day
        frailty = rng.normal(0, 0.3)        # hidden well quality
        t = DATA_START + pd.Timedelta(int(rng.integers(0, 120)), unit="D")
        history = []                         # end dates of past failures
        prev_run = np.nan
        while t < DATA_END:
            rpm = max(30.0, base_rpm + rng.normal(0, 20))
            wc = float(np.clip(base_wc + rng.normal(0, 0.05), 0, 0.99))
            oil = max(0.0, base_oil * rng.lognormal(0, 0.3))
            n_fail_365 = sum(1 for d in history if (t - d).days <= 365)
            repair = rng.choice(REPAIR_TYPES, p=[0.55, 0.25, 0.12, 0.08])
            # true hazard: higher rpm, water cut, recent failures -> shorter runs;
            # thorough repair -> longer runs
            lin = (0.004 * (rpm - 220) + 1.2 * (wc - 0.7) + 0.12 * min(n_fail_365, 6)
                   - 0.45 * (repair == "thorough") + frailty)
            scale = 180 * np.exp(-lin)       # Weibull scale in days
            dur = max(1, int(scale * rng.weibull(1.1)))
            end = t + pd.Timedelta(int(dur), unit="D")
            event = 1
            if end >= DATA_END:              # still running when data ends
                end, event = DATA_END, 0
                dur = max(1, (DATA_END - t).days)
            rows.append(dict(
                run_id=f"{uwi}-{len(history):03d}", uwi=uwi, facility=facility,
                start_date=t.normalize(), end_date=end.normalize(),
                duration_days=dur, event=event,
                rpm=round(rpm, 1), water_cut=round(wc, 3), oil_rate=round(oil, 2),
                n_fail_365=n_fail_365, prev_run_days=prev_run,
                repair_type=repair,
                            ))
            if event == 0:
                break
            history.append(end)
            prev_run = dur
            t = end + pd.Timedelta(int(rng.integers(1, 8)), unit="D")  # repair downtime
    df = pd.DataFrame(rows)
    for col in ["rpm", "water_cut", "oil_rate"]:
        df.loc[rng.random(len(df)) < 0.03, col] = np.nan
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/wells/runs.parquet")
    ap.add_argument("--wells", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    df = simulate(a.wells, a.seed)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(a.out, index=False)
    print(f"wrote {len(df):,} fake runs for {df.uwi.nunique():,} wells to {a.out}")
    print(f"failure share: {df.event.mean():.0%}, median run: {df.duration_days.median():.0f} days")


if __name__ == "__main__":
    main()
