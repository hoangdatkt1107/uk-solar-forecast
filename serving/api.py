""" to start:
    uvicorn serving.api:app --reload
    GET /                        -> the dashboard (index.html)
    GET /history.json            -> the 2-year backtest the dashboard loads
    GET /forecast?horizon=12h&model=stack

Static Web Apps isn't available in the deploy region, so the dashboard is served from
here (same origin, so the live overlay needs no CORS / API base). GRIDSIGHT_SERVE_DIR is
the Azure Files mount the serve-job writes to; GRIDSIGHT_FRONTEND_DIR holds the dashboard.
"""
from __future__ import annotations
import json
import os
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse

SERVE_DIR = Path(os.getenv("GRIDSIGHT_SERVE_DIR", "artifacts/serve"))
FRONTEND_DIR = Path(os.getenv("GRIDSIGHT_FRONTEND_DIR", "frontend"))
_ORIGINS = [o.strip() for o in os.getenv("GRIDSIGHT_CORS_ORIGINS", "*").split(",") if o.strip()]

app = FastAPI(title="GridSight UK Solar Forecast")
app.add_middleware(CORSMiddleware, allow_origins=_ORIGINS, allow_methods=["GET"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.get("/")
def index():
    f = FRONTEND_DIR / "index.html"
    if f.exists():
        return FileResponse(f)
    return {"service": "gridsight-uk-solar-forecast",
            "endpoints": ["/health", "/forecast?horizon=12h&model=stack"]}


@app.get("/history.json")
def history():
    f = FRONTEND_DIR / "history.json"
    if f.exists():
        return FileResponse(f, media_type="application/json")
    raise HTTPException(404, "history.json not found")


@app.get("/health")
def health():
    avail = sorted(p.stem.removeprefix("forecast_")
                   for p in SERVE_DIR.glob("forecast_*.json"))
    return {"status": "ok", "serve_dir": str(SERVE_DIR), "available": avail}


@app.get("/forecast")
def forecast(horizon: str = "12h", model: str = "stack"):
    for name in (f"forecast_{model}_{horizon}.json", f"forecast_{horizon}.json"):
        f = SERVE_DIR / name
        if f.exists():
            return json.loads(f.read_text())
    raise HTTPException(404, f"no forecast for model={model} horizon={horizon}; "
                             f"run the serve job (`python pipeline.py serve`)")


@app.get("/recent")
def recent(horizon: str = "12h", model: str = "stack", since: str | None = None):
    """Actual / NESO / model output for the recent past, refreshed by the hourly serve job
    (the stack's file keeps up to a year, see modeling/serve.py).

    history.json is a static backtest that only moves when it is rebuilt by hand, so the
    dashboard reads this to keep its `actual` line current in between. `since` (ISO time)
    drops the slots before it, so the dashboard only downloads what history.json lacks.
    """
    f = SERVE_DIR / f"recent_{model}_{horizon}.json"
    if not f.exists():
        raise HTTPException(404, f"no recent window for model={model} horizon={horizon}; "
                                 f"run the serve job (`python pipeline.py serve`)")
    payload = json.loads(f.read_text())
    if since:
        try:
            t0 = datetime.fromisoformat(since.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(422, f"since={since!r} is not an ISO timestamp")
        if t0.tzinfo is None:
            raise HTTPException(422, "since needs a timezone, e.g. 2026-07-15T00:00:00Z")
        payload["recent"] = [r for r in payload.get("recent") or []
                             if datetime.fromisoformat(r["timestamp_utc"]) >= t0]
        payload["n_points"] = len(payload["recent"])
        payload["n_actual"] = sum(1 for r in payload["recent"] if r.get("actual") is not None)
    return payload
