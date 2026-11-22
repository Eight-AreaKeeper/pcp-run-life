"""Steps 2 to 4: turn the three raw files into runs.parquet for train_eval.py.

    python src/build_runs.py --data data --out outputs

Reads   data/daily_production.parquet, data/well_jobs.parquet, data/production_failures.parquet
Writes  outputs/cleaning_log.csv           (aggregate, public)
        outputs/event_check.csv            (aggregate, public: match with the official failure log)
        outputs/wells/events.parquet       (well-level, gitignored)
        outputs/wells/runs.parquet         (well-level, gitignored; follows the README contract)

Definitions (write these in the README methods section)
- Wells: heavy-oil producers only (flow_direction = production, primary_fluid_type = Heavy Oil).
- Downhole day: hours_down > 0 and downtime_code_1 mentions "downhole"
  (rods, pump, tubing, rodstring, well torqued, unspecified downhole).
- Failure event: downhole days of one well, with gaps of MERGE_GAP days or less merged into one event.
- Producing day: hours_on_prod > 0 and not a downhole day.
- Run: starts on the first producing day after an event ends; ends at the start of the next event
  (event = 1) or at the well's last record (event = 0, still running / censored).
  duration_days counts producing days only, so long shut-ins do not count as run time.
- Features, all known on the restart day:
  rpm, water_cut, oil_rate  = over the last PRE_WINDOW producing days before the failure that started the run
  n_fail_365                = failure events of this well in the 365 days before the restart (incl. the one just fixed)
  prev_run_days             = length of the run that ended in that failure
  repair_type               = from well_jobs started within JOB_WINDOW days of the event:
                              thorough = any Workover job or a new rod string; patch = other Rod Truck job;
                              other = any other job; unknown = no job found
"""
import argparse
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

MERGE_GAP = 4        # days between downhole days that still count as the same event
PRE_WINDOW = 30      # producing days used for pre-failure features
JOB_WINDOW = 7       # days around an event to look for the repair job
LOG_WINDOW = 7       # days tolerance when matching events to the official failure log
RPM_MAX = 1000


def load_daily(path: Path, log: list) -> pd.DataFrame:
    con = duckdb.connect()
    total = con.execute(f"select count(*) from read_parquet('{path}')").fetchall()[0][0]
    df = con.sql(f"""
        select uwi, field_operating_area, facility_name, production_date as date,
               hours_down, hours_on_prod, downtime_code_1,
               oil_produced_m3 as oil, water_produced_m3 as water,
               cast(pcp_pump_rpm as double) as rpm
        from read_parquet('{path}')
        where flow_direction = 'production' and primary_fluid_type = 'Heavy Oil'
    """).df()
    log.append(("keep heavy-oil producers", "daily_production", total, len(df)))
    df["date"] = pd.to_datetime(df["date"])

    n = len(df)
    df = (df.sort_values(["uwi", "date", "hours_on_prod"], ascending=[True, True, False])
            .drop_duplicates(["uwi", "date"]))
    log.append(("drop duplicate well-days", "daily_production", n, len(df)))

    for col, cond, note in [
        ("rpm", lambda s: (s < 0) | (s > RPM_MAX), f"rpm < 0 or > {RPM_MAX} set to missing"),
        ("oil", lambda s: s < 0, "negative oil set to missing"),
        ("water", lambda s: s < 0, "negative water set to missing"),
    ]:
        bad = cond(df[col]).fillna(False)
        log.append((note, "daily_production", len(df), int(bad.sum())))
        df.loc[bad, col] = np.nan

    code = df["downtime_code_1"].fillna("")
    df["downhole"] = (df["hours_down"].fillna(0) > 0) & code.str.contains("downhole", case=False)
    df["producing"] = (df["hours_on_prod"].fillna(0) > 0) & ~df["downhole"]
    log.append(("downhole days found", "daily_production", len(df), int(df["downhole"].sum())))
    return df


def find_events(df: pd.DataFrame) -> pd.DataFrame:
    dh = df.loc[df["downhole"], ["uwi", "date", "downtime_code_1"]].sort_values(["uwi", "date"])
    gap = dh.groupby("uwi")["date"].diff().dt.days
    dh["event_no"] = ((gap.isna()) | (gap > MERGE_GAP + 1)).groupby(dh["uwi"]).cumsum()
    ev = (dh.groupby(["uwi", "event_no"])
            .agg(event_start=("date", "min"), event_end=("date", "max"),
                 down_days=("date", "size"),
                 downtime_code=("downtime_code_1", lambda s: s.mode().iat[0]))
            .reset_index())
    return ev


def classify_jobs(jobs: pd.DataFrame) -> pd.DataFrame:
    jobs = jobs.copy()
    jt = jobs["job_type"].fillna("")
    cat = jobs["job_category"].fillna("")
    jobs["repair_class"] = np.select(
        [cat.eq("Workover") | jt.str.contains("New String", case=False),
         cat.eq("Rod Truck")],
        ["thorough", "patch"], default="other")
    jobs["start_date"] = pd.to_datetime(jobs["start_date"])
    return jobs


def attach_jobs(ev: pd.DataFrame, jobs: pd.DataFrame) -> pd.DataFrame:
    m = ev[["uwi", "event_no", "event_start", "event_end"]].merge(
        jobs[["uwi", "start_date", "repair_class", "job_type", "field_est_total"]], on="uwi")
    m = m[(m["start_date"] >= m["event_start"] - pd.Timedelta(days=JOB_WINDOW)) &
          (m["start_date"] <= m["event_end"] + pd.Timedelta(days=JOB_WINDOW))]
    rank = {"thorough": 0, "patch": 1, "other": 2}
    m["r"] = m["repair_class"].map(rank)
    best = (m.sort_values("r").groupby(["uwi", "event_no"])
              .agg(repair_type=("repair_class", "first"), job_type=("job_type", "first"),
                   est_cost=("field_est_total", "sum"), n_jobs=("r", "size"))
              .reset_index())
    ev = ev.merge(best, on=["uwi", "event_no"], how="left")
    ev["repair_type"] = ev["repair_type"].fillna("unknown")
    return ev


def build_runs(df: pd.DataFrame, ev: pd.DataFrame) -> pd.DataFrame:
    rows = []
    prod = df[df["producing"]][["uwi", "date", "oil", "water", "rpm"]]
    prod_by_well = {u: g.reset_index(drop=True) for u, g in prod.groupby("uwi")}
    last_date = df.groupby("uwi")["date"].max()
    meta = df.groupby("uwi")[["field_operating_area", "facility_name"]].agg(
        lambda s: s.mode().iat[0] if s.notna().any() else None)

    for uwi, e in ev.sort_values(["uwi", "event_start"]).groupby("uwi"):
        p = prod_by_well.get(uwi)
        if p is None or p.empty:
            continue
        dates = p["date"].values
        e = e.reset_index(drop=True)
        prev_len = np.nan
        for k, r in e.iterrows():
            # restart = first producing day after this event
            i0 = np.searchsorted(dates, np.datetime64(r["event_end"]), side="right")
            if i0 >= len(dates):
                break
            start = pd.Timestamp(dates[i0])
            if k + 1 < len(e):
                nxt = e.loc[k + 1, "event_start"]
                if start >= nxt:          # failed again before producing: fold into next event
                    continue
                i1 = np.searchsorted(dates, np.datetime64(nxt), side="left")
                event, end = 1, nxt
            else:
                i1, event, end = len(dates), 0, last_date[uwi]
            dur = int(i1 - i0)
            if dur < 1:
                continue
            # features from the PRE_WINDOW producing days before the failure that started this run
            j1 = np.searchsorted(dates, np.datetime64(r["event_start"]), side="left")
            w = p.iloc[max(0, j1 - PRE_WINDOW):j1]
            oil_sum, water_sum = w["oil"].sum(), w["water"].sum()
            es = e["event_start"].values.astype("datetime64[ns]")
            t0 = np.datetime64(start.to_datetime64(), "ns")
            fails_365 = int(((es < t0) & (es >= t0 - np.timedelta64(365, "D"))).sum())
            rows.append(dict(
                run_id=f"{uwi}-{k:03d}", uwi=uwi,
                field_operating_area=meta.loc[uwi, "field_operating_area"],
                facility=meta.loc[uwi, "facility_name"],
                failure_start=r["event_start"], start_date=start, end_date=pd.Timestamp(end),
                duration_days=dur, event=event,
                rpm=w["rpm"].mean() if len(w) else np.nan,
                water_cut=(water_sum / (oil_sum + water_sum)) if (oil_sum + water_sum) > 0 else np.nan,
                oil_rate=w["oil"].mean() if len(w) else np.nan,
                n_fail_365=fails_365,
                prev_run_days=prev_len,
                repair_type=r["repair_type"],
                repair_job_type=r.get("job_type"),
                repair_est_cost=r.get("est_cost"),
                downtime_code=r["downtime_code"],
            ))
            prev_len = dur if event == 1 else np.nan
    return pd.DataFrame(rows)


def check_events(ev: pd.DataFrame, fails: pd.DataFrame, wells: set) -> pd.DataFrame:
    f = fails[fails["uwi"].isin(wells)].copy()
    f["failure_start_date"] = pd.to_datetime(f["failure_start_date"])
    log_start = f["failure_start_date"].min()
    e = ev[ev["event_start"] >= log_start]
    tol = pd.Timedelta(days=LOG_WINDOW)

    m = f[["uwi", "failure_start_date"]].merge(e[["uwi", "event_start", "event_end"]], on="uwi")
    hit = m[(m["failure_start_date"] >= m["event_start"] - tol) &
            (m["failure_start_date"] <= m["event_end"] + tol)]
    log_found = hit.drop_duplicates(["uwi", "failure_start_date"]).shape[0]
    ev_found = hit.drop_duplicates(["uwi", "event_start"]).shape[0]
    return pd.DataFrame([
        dict(check="official log failures matched by an event (recall)",
             matched=log_found, total=len(f), share=round(log_found / max(len(f), 1), 3)),
        dict(check="events since log start matched by a log entry (precision)",
             matched=ev_found, total=len(e), share=round(ev_found / max(len(e), 1), 3)),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()
    data, out = Path(a.data), Path(a.out)
    (out / "wells").mkdir(parents=True, exist_ok=True)
    log = []

    df = load_daily(data / "daily_production.parquet", log)
    jobs = classify_jobs(pd.read_parquet(data / "well_jobs.parquet"))
    fails = pd.read_parquet(data / "production_failures.parquet")

    ev = attach_jobs(find_events(df), jobs)
    log.append(("failure events (downhole days merged)", "events", int(df["downhole"].sum()), len(ev)))
    ev.to_parquet(out / "wells" / "events.parquet", index=False)

    chk = check_events(ev, fails, set(df["uwi"]))
    chk.to_csv(out / "event_check.csv", index=False)

    runs = build_runs(df, ev)
    log.append(("runs built", "runs", len(ev), len(runs)))
    runs.to_parquet(out / "wells" / "runs.parquet", index=False)

    pd.DataFrame(log, columns=["step", "table", "rows_before", "rows_after_or_affected"]).to_csv(
        out / "cleaning_log.csv", index=False)

    print(f"events: {len(ev):,} on {ev['uwi'].nunique():,} wells")
    print(chk.to_string(index=False))
    print(f"runs: {len(runs):,} on {runs['uwi'].nunique():,} wells, "
          f"{runs['event'].mean():.0%} end in failure, median {runs['duration_days'].median():.0f} producing days")
    print("repair_type:", runs["repair_type"].value_counts().to_dict())
    print(f"wrote {out}/wells/runs.parquet, {out}/wells/events.parquet, {out}/cleaning_log.csv, {out}/event_check.csv")


if __name__ == "__main__":
    main()
