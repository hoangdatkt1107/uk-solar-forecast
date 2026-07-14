"""Generate the dashboard's historical series across the full out-of-sample range
(2024-07 onward — training was Jan-Jun 2024), for BOTH models:

  * stack   — TCN-Q + LGBM-Q, quantile (uses NWP weather)
  * chronos — amazon/chronos-bolt-base, univariate zero-shot (target history only)

joined with the actuals + NESO baseline + weather context from Gold. Columnar +
integer-rounded, weather stored display-ready, to keep the file small.

    KMP_DUPLICATE_LIB_OK=TRUE python frontend/build_history.py   # writes frontend/history.json

The KMP_DUPLICATE_LIB_OK=TRUE guard avoids an OpenMP double-load segfault when the
LightGBM (stack) and Chronos torch runtimes coexist in one process. Chronos runs on CPU
for the same reason. Takes ~8 min (stack + chronos, both horizons).

Structure:
  series["12h"|"6h"] = {start, step_min, n,
                        actual, neso, ssrd, cloud, temp, wind, day,   # shared context
                        models: {stack:{q10,q50,q90}, chronos:{q10,q50,q90}}}
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modeling.predict import predict_gold
from modeling.config import ModelConfig

TS = "timestamp_utc"
OUT_START = "2024-07-01"          # first fully out-of-sample day
STEP = pd.Timedelta(minutes=30)
SEQ_CTX = pd.Timedelta(hours=96)  # >= seq_len*30min of context for the stack chunks
CHRONOS_CTX = 512                 # chronos context length


def _load_gold(hours: str) -> pd.DataFrame:
    files = sorted(Path(f"data/gold/gold_features_{hours}").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df[TS] = pd.to_datetime(df[TS], utc=True)
    return df.sort_values(TS).reset_index(drop=True)


# stack: predict over the whole range, quarter by quarter (no boundary gaps)
def _stack_full(gold: pd.DataFrame, artifacts_dir: str) -> pd.DataFrame:
    start, end = pd.Timestamp(OUT_START, tz="UTC"), gold[TS].max()
    out, q = [], start
    while q <= end:
        q_end = min((q + pd.Timedelta(days=92)).normalize(), end + STEP)
        chunk = gold[(gold[TS] >= q - SEQ_CTX) & (gold[TS] < q_end)]
        if len(chunk) > 130:
            p = predict_gold(chunk, artifacts_dir)
            p[TS] = pd.to_datetime(p[TS], utc=True)
            out.append(p[(p[TS] >= q) & (p[TS] < q_end)])
        q = q_end
    r = pd.concat(out, ignore_index=True)
    return r.rename(columns={"pred_q10": "s_q10", "pred_q50": "s_q50", "pred_q90": "s_q90"})


# chronos: univariate zero-shot, one fixed-lead forecast per daylight slot 
def _chronos_full(gold: pd.DataFrame, horizon: int, batch: int = 256) -> pd.DataFrame:
    import torch
    from chronos import BaseChronosPipeline
    # CPU on purpose: the stack (predict_gold) already used torch on CPU in this process,
    pipe = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-bolt-base", device_map="cpu", torch_dtype=torch.float32)
    y = gold["target_cf"].to_numpy("float32")
    cap = gold["capacity_mwp"].to_numpy("float32") if "capacity_mwp" in gold else np.ones(len(gold), "float32")
    ts = gold[TS].to_numpy()
    day = (gold["is_daylight"] == 1).to_numpy() if "is_daylight" in gold else np.ones(len(gold), bool)
    inrange = (gold[TS] >= pd.Timestamp(OUT_START, tz="UTC")).to_numpy()
    idx = np.where(day & inrange)[0]
    idx = idx[idx - horizon - CHRONOS_CTX >= 0]
    idx = idx[np.isfinite(y[idx])]
    qs = [0.1, 0.5, 0.9]
    rows_ts, cq = [], {q: [] for q in qs}
    for s in range(0, len(idx), batch):
        chunk = idx[s:s + batch]
        ctx = [torch.tensor(np.nan_to_num(y[i - horizon - CHRONOS_CTX + 1: i - horizon + 1]))
               for i in chunk]
        q, _ = pipe.predict_quantiles(ctx, prediction_length=horizon, quantile_levels=qs)
        last = q[:, -1, :].cpu().numpy()
        for k, ci in enumerate(chunk):
            rows_ts.append(ts[ci])
            for j, ql in enumerate(qs):
                cq[ql].append(float(last[k, j]) * cap[ci])
        print(f"  chronos {min(s + batch, len(idx))}/{len(idx)}")
    return pd.DataFrame({TS: pd.to_datetime(rows_ts, utc=True),
                         "c_q10": cq[0.1], "c_q50": cq[0.5], "c_q90": cq[0.9]})


def _horizon(hours: str, horizon: int, artifacts_dir: str) -> dict:
    gold = _load_gold(hours)
    stack = _stack_full(gold, artifacts_dir)
    chronos = _chronos_full(gold, horizon)

    ctx = gold[[TS, "target_mw", "embedded_solar_mw", "ssrd_uk", "tcc_uk", "t2m_uk", "ws10_uk", "is_daylight"]]
    df = ctx.merge(stack, on=TS, how="left").merge(chronos, on=TS, how="left")

    grid = pd.date_range(pd.Timestamp(OUT_START, tz="UTC"), df[TS].max(), freq="30min", tz="UTC", name=TS)
    df = df.set_index(TS).reindex(grid).reset_index()
    day = np.where(pd.to_numeric(df["is_daylight"], errors="coerce").fillna(0) >= 0.5, 1, 0)

    def icol(name, scale=1.0, off=0.0):
        v = pd.to_numeric(df[name], errors="coerce") * scale + off
        return [None if pd.isna(x) else int(round(x)) for x in v]

    def qcol(name):                                   # night forced to 0, else model value
        v = pd.to_numeric(df[name], errors="coerce")
        return [0 if day[k] == 0 else (None if pd.isna(x) else int(round(x)))
                for k, x in enumerate(v)]

    return {
        "start": grid.min().isoformat(), "step_min": 30, "n": len(df),
        "actual": icol("target_mw"), "neso": icol("embedded_solar_mw"),
        "ssrd": icol("ssrd_uk"),
        "cloud": icol("tcc_uk", 100.0),               # fraction -> %
        "temp": icol("t2m_uk", 1.0, -273.15),         # K -> °C
        "wind": icol("ws10_uk", 10.0),                # m/s ×10 
        "day": [int(d) for d in day],
        "models": {
            "stack": {"q10": qcol("s_q10"), "q50": qcol("s_q50"), "q90": qcol("s_q90")},
            "chronos": {"q10": qcol("c_q10"), "q50": qcol("c_q50"), "q90": qcol("c_q90")},},
            }


def main() -> None:
    series = {
        "12h": _horizon("12h", 24, "artifacts/model_12h"),
        "6h": _horizon("6h", 12, "artifacts/model_6h"),
    }
    payload = {"generated_at": pd.Timestamp.now("UTC").isoformat(), "series": series}
    out = Path("frontend/history.json")
    out.write_text(json.dumps(payload, separators=(",", ":")))
    for k, s in series.items():
        print(f"{k}: {s['n']} pts  {s['start'][:10]}..  models={list(s['models'])}")
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
