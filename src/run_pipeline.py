"""
run_pipeline.py
----------------------------------------------------------------------------
Master runner for the FI-AdaBoost solar energy forecasting thesis pipeline.

Pipeline stages (run in order):
  1. data_acquisition.py   - fetch 3,000 NASA POWER points + OSM buildings
  2. data_processing.py    - clean raw CSVs / GeoJSONs
  3. feature_engineering.py- compute per-building topographical features
  4. data_integration.py   - spatial nearest-neighbour join (3,000 pts + OSM)
  5. model_training.py     - train Baseline AdaBoost & FI-AdaBoost, evaluate

Usage:
  python run_pipeline.py              # full pipeline (start from stage 1)
  python run_pipeline.py --from 3     # resume from feature_engineering
  python run_pipeline.py --from 5     # model training only (data already built)
  python run_pipeline.py --check      # verify all required files exist, then exit
----------------------------------------------------------------------------
"""
import os
import sys
import argparse
import time

# -- Directory layout ----------------------------------------------------------
# ROOT_DIR is the parent of the folder containing this file
# (matches the os.path.dirname(os.path.dirname(...)) logic in every module)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.dirname(SCRIPT_DIR)

RAW_DIR       = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
RESULTS_DIR   = os.path.join(ROOT_DIR, "results")

YEAR      = "2024"
CITY_SLUG = "davao_city"


# -- Expected files per stage --------------------------------------------------
STAGE_OUTPUTS = {
    1: [
        os.path.join(RAW_DIR, f"baseline_spatial_dataset_{CITY_SLUG}_{YEAR}.csv"),
        os.path.join(RAW_DIR, "osm_buildings.geojson"),
    ],
    2: [
        os.path.join(PROCESSED_DIR, f"baseline_spatial_clean_{YEAR}.csv"),
        os.path.join(PROCESSED_DIR, "osm_clean.geojson"),
    ],
    3: [
        os.path.join(PROCESSED_DIR, "osm_features.geojson"),
    ],
    4: [
        os.path.join(PROCESSED_DIR, "integrated_dataset.csv"),
    ],
    5: [
        os.path.join(RESULTS_DIR, "table1_training.csv"),
        os.path.join(RESULTS_DIR, "table2_test.csv"),
        os.path.join(RESULTS_DIR, "improvement_summary.csv"),
    ],
}

STAGE_NAMES = {
    1: "Data Acquisition  [NASA POWER + OSM]",
    2: "Data Processing   [clean raw files]",
    3: "Feature Engineering [OSM topographical features]",
    4: "Data Integration  [spatial nearest-neighbour join]",
    5: "Model Training    [Baseline AdaBoost vs FI-AdaBoost]",
}


# -- Helpers -------------------------------------------------------------------

def banner(text, char="=", width=74):
    print(f"\n{char * width}")
    print(f"  {text}")
    print(f"{char * width}")


def check_stage_outputs(stage: int) -> bool:
    """Return True if all expected outputs for a stage already exist."""
    return all(os.path.exists(p) for p in STAGE_OUTPUTS[stage])


def print_file_status(paths):
    for p in paths:
        exists = os.path.exists(p)
        icon   = "[OK]" if exists else "[--]"
        size   = f"  ({os.path.getsize(p):,} bytes)" if exists else ""
        print(f"    {icon}  {os.path.basename(p)}{size}")


def check_dependencies():
    """Verify all required packages are importable."""
    required = {
        "numpy":      "numpy",
        "pandas":     "pandas",
        "requests":   "requests",
        "geopandas":  "geopandas",
        "osmnx":      "osmnx",
        "scipy":      "scipy",
        "sklearn":    "scikit-learn",
        "matplotlib": "matplotlib",
        "shapely":    "shapely",
    }
    missing = []
    for mod, pkg in required.items():
        try:
            __import__(mod)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"[ERROR] Missing packages: {missing}")
        print(f"  Install with:  pip install {' '.join(missing)}")
        sys.exit(1)
    print("[Dependencies] All required packages found.")


# -- Stage runners -------------------------------------------------------------

def run_stage_1():
    """Data acquisition - fetch NASA POWER + OSM buildings."""
    import data_acquisition as da
    print("\n  NOTE: Stage 1 fetches 3,000 NASA POWER API calls.")
    print("  Checkpoint saves every 50 rows - safe to interrupt and resume.")
    print("  Estimated time: 15–30 minutes depending on API speed.\n")
    da.fetch_nasa_baseline_spatial(
        place_name="Davao City, Philippines",
        year=YEAR,
        n_points=3000,
        seed=42,
    )
    da.fetch_osm_data(place_name="Davao City, Philippines")


def run_stage_2():
    """Data processing - clean raw spatial CSV and OSM GeoJSON."""
    import data_processing as dp
    baseline_df = dp.process_baseline_spatial(year=YEAR)
    baseline_df.to_csv(
        os.path.join(PROCESSED_DIR, f"baseline_spatial_clean_{YEAR}.csv"),
        index=False,
    )
    print(f"  Saved baseline_spatial_clean_{YEAR}.csv  ({len(baseline_df)} rows)")

    osm_gdf = dp.process_osm()
    osm_gdf.to_file(
        os.path.join(PROCESSED_DIR, "osm_clean.geojson"),
        driver="GeoJSON",
    )
    print(f"  Saved osm_clean.geojson  ({len(osm_gdf):,} buildings)")


def run_stage_3():
    """Feature engineering - compute OSM topographical features."""
    import geopandas as gpd
    import feature_engineering as fe

    osm_path = os.path.join(PROCESSED_DIR, "osm_clean.geojson")
    osm_gdf  = gpd.read_file(osm_path)
    print(f"  Loaded {len(osm_gdf):,} buildings from osm_clean.geojson")

    osm_gdf = fe.topo_features(osm_gdf)
    osm_gdf = fe.normalize_sei(osm_gdf)

    out = os.path.join(PROCESSED_DIR, "osm_features.geojson")
    osm_gdf.to_file(out, driver="GeoJSON")
    print(f"  Saved osm_features.geojson  ({len(osm_gdf):,} buildings)")


def run_stage_4():
    """Data integration - spatial nearest-neighbour join."""
    import data_integration as di
    di.integrate_datasets()


def run_stage_5():
    """Model training - Baseline AdaBoost vs FI-AdaBoost."""
    import model_training as mt
    mt.main()


STAGE_RUNNERS = {
    1: run_stage_1,
    2: run_stage_2,
    3: run_stage_3,
    4: run_stage_4,
    5: run_stage_5,
}


# -- Main ----------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the FI-AdaBoost solar forecasting pipeline."
    )
    parser.add_argument(
        "--from", dest="start_stage", type=int, default=1,
        metavar="N", help="Start pipeline from stage N (1–5). Default: 1."
    )
    parser.add_argument(
        "--check", action="store_true",
        help="Print file status for all stages then exit."
    )
    args = parser.parse_args()

    # -- Create directories ------------------------------------------------
    for d in (RAW_DIR, PROCESSED_DIR, RESULTS_DIR):
        os.makedirs(d, exist_ok=True)

    # -- sys.path: ensure script directory is importable -------------------
    if SCRIPT_DIR not in sys.path:
        sys.path.insert(0, SCRIPT_DIR)

    # -- Dependency check --------------------------------------------------
    check_dependencies()

    # -- --check mode: just print status ----------------------------------
    if args.check:
        banner("Pipeline File Status Check")
        for stage in range(1, 6):
            ok = check_stage_outputs(stage)
            status = "COMPLETE" if ok else "PENDING"
            print(f"\n  Stage {stage} - {STAGE_NAMES[stage]}  [{status}]")
            print_file_status(STAGE_OUTPUTS[stage])
        return

    # -- Run pipeline ------------------------------------------------------
    banner(
        f"FI-AdaBoost Solar Forecasting Pipeline  -  Davao City {YEAR}\n"
        f"  Starting from stage {args.start_stage}"
    )

    for stage in range(args.start_stage, 6):
        banner(f"STAGE {stage} - {STAGE_NAMES[stage]}", char="-")

        # Skip if outputs already exist (and we're not rerunning from this stage explicitly)
        if args.start_stage < stage and check_stage_outputs(stage):
            print(f"  [SKIP] Stage {stage} outputs already exist.")
            print_file_status(STAGE_OUTPUTS[stage])
            continue

        t0 = time.time()
        try:
            STAGE_RUNNERS[stage]()
        except FileNotFoundError as e:
            print(f"\n[ERROR] Missing input for Stage {stage}:\n  {e}")
            print(f"  Run the pipeline from stage {stage - 1} first.")
            sys.exit(1)
        except Exception as e:
            print(f"\n[ERROR] Stage {stage} failed: {e}")
            raise

        elapsed = time.time() - t0
        print(f"\n  Stage {stage} complete in {elapsed:.1f}s")
        print("  Output files:")
        print_file_status(STAGE_OUTPUTS[stage])

    banner("Pipeline Complete")
    print(f"  Results saved to: {RESULTS_DIR}")
    print()


if __name__ == "__main__":
    main()
