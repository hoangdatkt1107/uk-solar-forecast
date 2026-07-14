"""NESO Data Portal (CKAN) -> Bronze parquet"""
from __future__ import annotations
import json
import re
import shutil
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

from .common import BRONZE_LOCAL_DIR

NESO_API = "https://api.neso.energy/api/3/action"
# GridSight-relevant default datasets
NESO_DEFAULT_PACKAGES = ["embedded-wind-and-solar-forecasts"]
_CKAN_MIN_INTERVAL = 1.1 


def _neso_dir(package: str) -> Path:
    d = BRONZE_LOCAL_DIR / "neso" / package
    d.mkdir(parents=True, exist_ok=True)
    return d

def _slug(name: str) -> str:
    s = re.sub(r"[^0-9A-Za-z]+", "_", str(name)).strip("_").lower()
    return s or "resource"

def _load_state(package: str) -> dict:
    f = _neso_dir(package) / "_state.json"
    if f.exists():
        try:
            return json.loads(f.read_text())
        except Exception:
            return {}
    return {}

def _save_state(package: str, state: dict) -> None:
    (_neso_dir(package) / "_state.json").write_text(json.dumps(state, indent=2))


class _Throttle:
    """Simple min-interval gate so we never exceed the CKAN rate limit."""

    def __init__(self, interval: float):
        self.interval = interval
        self._last = 0.0

    def wait(self) -> None:
        dt = time.monotonic() - self._last
        if dt < self.interval:
            time.sleep(self.interval - dt)
        self._last = time.monotonic()


def _resource_year(name: str) -> int | None:
    m = re.search(r"(19|20)\d{2}", str(name))
    return int(m.group(0)) if m else None


def _download_csv_to_parquet(client, resource: dict, out_path: Path,
                             throttle: _Throttle, max_records: int | None) -> int:
    """Stream a CSV resource to temp, convert to parquet via pyarrow, return row count."""
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    url = resource["url"]
    tmp = Path(tempfile.mkdtemp(prefix="neso_")) / "data.csv"
    try:
        throttle.wait()
        with client.stream("GET", url, timeout=600, follow_redirects=True) as r:
            r.raise_for_status()
            with open(tmp, "wb") as fh:
                for chunk in r.iter_bytes(chunk_size=1 << 20):  # 1 MB chunks
                    fh.write(chunk)

        read_opts = pacsv.ReadOptions(block_size=1 << 24)
        table = pacsv.read_csv(tmp, read_options=read_opts)
        if max_records is not None and table.num_rows > max_records:
            table = table.slice(0, max_records)

        n = table.num_rows
        now = datetime.now(timezone.utc).isoformat()
        table = table.append_column("fetched_at", pa.array([now] * n, pa.string()))
        table = table.append_column("source_url", pa.array([url] * n, pa.string()))
        table = table.append_column("resource_id", pa.array([resource["id"]] * n, pa.string()))
        pq.write_table(table, out_path, compression="snappy")
        return n
    finally:
        shutil.rmtree(tmp.parent, ignore_errors=True)

def ingest_neso(
    packages: list[str] | None = None,
    years: list[int] | None = None,
    force_refresh: bool = False,
    max_records: int | None = None,
) -> None:

    try:
        import httpx
        import pyarrow  
    except ImportError as e:
        logger.error(f"missing dep: {e}. pip install httpx pyarrow")
        return

    packages = packages or NESO_DEFAULT_PACKAGES
    want_years = {int(y) for y in years} if years else None
    throttle = _Throttle(_CKAN_MIN_INTERVAL)
    logger.info(
        f"NESO ingest | packages={packages} years={sorted(want_years) if want_years else 'all'} "
        f"force_refresh={force_refresh}"
    )

    with httpx.Client(headers={"User-Agent": "gridsight-bronze/1.0"}) as client:
        for package in packages:
            logger.info(f"--- package: {package} ---")
            try:
                throttle.wait()
                r = client.get(f"{NESO_API}/package_show", params={"id": package}, timeout=60)
                r.raise_for_status()
                resources = r.json()["result"].get("resources", [])
            except Exception as e:
                logger.error(f"package_show failed for {package}: {e}")
                continue

            state = _load_state(package)
            done = skipped = failed = 0
            for res in resources:
                if str(res.get("format", "")).upper() != "CSV":
                    continue  # skip docs / non-tabular files
                name, rid = res.get("name", res["id"]), res["id"]

                yr = _resource_year(name)
                if want_years is not None and yr is not None and yr not in want_years:
                    continue  # archive year not requested

                out_path = _neso_dir(package) / f"{_slug(name)}.parquet"
                last_mod = res.get("last_modified") or res.get("created")
                if (not force_refresh and out_path.exists()
                        and state.get(rid) == last_mod and out_path.stat().st_size > 256):
                    logger.info(f"skip {name} (unchanged)")
                    skipped += 1
                    continue

                logger.info(f"fetching {name} -> {out_path.name}")
                try:
                    n = _download_csv_to_parquet(client, res, out_path, throttle, max_records)
                    state[rid] = last_mod
                    _save_state(package, state)
                    done += 1
                    logger.success(f"wrote {n:,} rows -> {out_path}")
                except Exception as e:
                    failed += 1
                    logger.error(f"  failed {name}: {e}")

            logger.success(
                f"NESO {package} complete: written={done} skipped={skipped} failed={failed} "
                f"-> {_neso_dir(package)}"
            )
