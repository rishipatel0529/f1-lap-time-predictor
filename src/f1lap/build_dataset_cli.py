"""
src/f1lap/build_dataset_cli.py

Command-line helper to (re)build `data/all_telemetry_track_data.csv` end-to-end.
This orchestrates your legacy scripts to fetch driver telemetry, derive track
features, pull weather, and finally merge everything into the single training CSV.
Behavior is driven by `config/etl.yaml`; if merge fails, it falls back to the
most recently found “all_telemetry…csv” in `data/`.
"""

from __future__ import annotations
import argparse, subprocess, sys, yaml, glob
from pathlib import Path

# Run a subprocess while echoing the exact command; return its exit status
def _run(cmd): print("[run]", " ".join(cmd)); return subprocess.call(cmd)

def maybe_call(pyfile, args):
    if pyfile and Path(pyfile).exists(): return _run([sys.executable, pyfile] + args)
    print(f"[skip] {pyfile} (not found)"); return 0

def main():
    # Parse CLI, load ETL config, and extract path/script sections
    ap = argparse.ArgumentParser(description="Rebuild data/all_telemetry_track_data.csv via legacy scripts")
    ap.add_argument("--config", default="config/etl.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, "r"))
    paths, scripts = cfg["paths"], cfg["scripts"]

    # Ensure expected directory structure exists before running any producers
    Path(paths["driver_csv_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["track_data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["weather_dir"]).mkdir(parents=True, exist_ok=True)
    Path(paths["data_root"]).mkdir(parents=True, exist_ok=True)

    # Driver/telemetry fetchers (optional if you already have CSVs)
    maybe_call(scripts.get("fetch_fastf1"), [])
    maybe_call(scripts.get("fetch_historical_fastf1"), [])
    maybe_call(scripts.get("fetch_f1tv"), [])

    # Track and Weather
    maybe_call(scripts.get("track_features"), [])
    maybe_call(scripts.get("fetch_weather"), [])

    # Merge
    out_csv = Path(paths["output_csv"]); out_csv.parent.mkdir(parents=True, exist_ok=True)
    rc = maybe_call(scripts.get("merge"), ["--output", str(out_csv)])
    if rc != 0 or not out_csv.exists():
        # try to find a recent merge-like file and adopt it
        cands = glob.glob("data/*all*telemetry*track*data*.csv") + glob.glob("data/*merged*.csv")
        if cands:
            cand = max(cands, key=lambda p: Path(p).stat().st_mtime)
            print(f"[fallback] using existing: {cand}")
            Path(cand).replace(out_csv)
        else:
            print("[error] merge failed and no fallback CSV found"); sys.exit(1)

    print(f"[done] built {out_csv}")

if __name__ == "__main__":
    main()
