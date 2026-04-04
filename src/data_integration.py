"""
data_integration.py
────────────────────────────────────────────────────────────────────────────
Integrates the 3,000-point spatial dataset with per-building OSM features.

DESIGN
──────
  Input A : baseline_spatial_clean_2024.csv
              3,000 rows  —  lat | lon | GHI_mean_2024 (kWh/m²/day)
              (one row per random coordinate inside Davao City)

  Input B : osm_features.geojson
              up to 10,000 building polygons with computed features:
              rooftop_area_sq_m, orientation_score, shading_factor,
              tilt_factor, solar_exposure_index, SEI_norm

  Join method : Spatial nearest-neighbour
              Each of the 3,000 random points is matched to the
              geographically closest OSM building centroid.
              The matched building's features are attached to the point.

  Why nearest-neighbour (not a cross-join):
              Each coordinate represents ONE location in the city.
              The most appropriate building features for that location
              are those of the nearest building — just as an on-site
              survey would measure the rooftop at that address.

  Output : integrated_dataset.csv  —  3,000 rows
              lat | lon | GHI_mean_J | rooftop_area_sq_m |
              orientation_score | shading_factor | tilt_factor | SEI_norm

  Unit conversion:
              GHI_mean_kWh × 3,600,000 = GHI_mean_J  (J/m²/day)
              This matches the Quezon City baseline study's unit for
              RMSE and MAE (their Table 2 values are in J/m²).
────────────────────────────────────────────────────────────────────────────
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree

ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")

KWH_TO_J  = 3_600_000   # 1 kWh = 3,600,000 J
YEAR      = "2024"

# Building feature columns to carry into the integrated dataset
BUILDING_COLS = [
    "rooftop_area_sq_m",
    "orientation_score",
    "shading_factor",
    "tilt_factor",
    "SEI_norm",
]


def integrate_datasets() -> None:
    """
    §2.1.3.3 Data Integration and Consistency Check.
    Joins the 3,000-point NASA spatial dataset with per-building OSM features
    using a spatial nearest-neighbour match (cKDTree). Each random coordinate
    is assigned the features of its geographically closest OSM building,
    replicating the per-address survey approach described in §2.1.3.3:
    "every building polygon will have assigned to it values for NASA POWER
    parameters based upon its latitude and longitude."
    Also converts GHI kWh/m²/day → J/m²/day to match Quezon City study units.
    Output: integrated_dataset.csv — 3,000 rows ready for model_training.py.
    """
    print("=" * 60)
    print("Data Integration — Spatial Nearest-Neighbour Join")
    print("=" * 60)

    # ── Load inputs ───────────────────────────────────────────────────────
    spatial_path = os.path.join(PROCESSED_DIR,
                                f"baseline_spatial_clean_{YEAR}.csv")
    osm_path     = os.path.join(PROCESSED_DIR, "osm_features.geojson")

    if not os.path.exists(spatial_path):
        raise FileNotFoundError(
            f"Missing: {spatial_path}\n"
            "Run data_processing.process_baseline_spatial() first."
        )
    if not os.path.exists(osm_path):
        raise FileNotFoundError(
            f"Missing: {osm_path}\n"
            "Run feature_engineering.py first."
        )

    spatial_df = pd.read_csv(spatial_path)
    osm_gdf    = gpd.read_file(osm_path)

    target_col = f"GHI_mean_{YEAR}"
    if target_col not in spatial_df.columns:
        raise ValueError(f"Column '{target_col}' not found in {spatial_path}.")

    print(f"[Integration] Spatial points : {len(spatial_df):,}")
    print(f"[Integration] OSM buildings  : {len(osm_gdf):,}")

    # ── Validate / recompute building features ────────────────────────────
    missing_cols = [c for c in BUILDING_COLS if c not in osm_gdf.columns]
    if missing_cols:
        print(f"[Integration] Missing OSM columns: {missing_cols} — "
              "re-running feature engineering …")
        from feature_engineering import topo_features, normalize_sei
        osm_gdf = topo_features(osm_gdf)
        osm_gdf = normalize_sei(osm_gdf)

    # ── Project OSM to WGS84 for coordinate matching ─────────────────────
    if osm_gdf.crs and osm_gdf.crs.to_epsg() != 4326:
        osm_gdf = osm_gdf.to_crs("EPSG:4326")

    # ── Build cKDTree on OSM building centroids ───────────────────────────
    centroids     = osm_gdf.geometry.centroid
    osm_coords    = np.column_stack([centroids.y, centroids.x])  # lat, lon
    tree          = cKDTree(osm_coords)

    # ── Query: for each spatial point find its nearest building ──────────
    query_coords  = spatial_df[["lat", "lon"]].values
    distances, idx = tree.query(query_coords, k=1)

    print(f"[Integration] Nearest-neighbour matching done.")
    print(f"  Max distance to nearest building : {distances.max()*111:.2f} km")
    print(f"  Mean distance                    : {distances.mean()*111:.3f} km")

    # ── Attach building features to each spatial point ────────────────────
    matched_bldg = osm_gdf.iloc[idx][BUILDING_COLS].reset_index(drop=True)
    integrated   = pd.concat(
        [spatial_df.reset_index(drop=True), matched_bldg],
        axis=1,
    )

    # ── Convert GHI_mean: kWh/m²/day → J/m²/day ─────────────────────────
    integrated["GHI_mean_J"] = integrated[target_col] * KWH_TO_J

    # ── Drop rows with any NaN in critical columns ────────────────────────
    critical = ["lat", "lon", target_col, "GHI_mean_J"] + BUILDING_COLS
    n_before = len(integrated)
    integrated = (integrated
                  .dropna(subset=critical)
                  .reset_index(drop=True))
    n_dropped = n_before - len(integrated)
    if n_dropped:
        print(f"[Integration] Dropped {n_dropped} rows with NaN values.")

    # ── Select and order final columns ────────────────────────────────────
    final_cols = ["lat", "lon", target_col, "GHI_mean_J"] + BUILDING_COLS
    integrated = integrated[final_cols]

    # ── Save ──────────────────────────────────────────────────────────────
    out_path = os.path.join(PROCESSED_DIR, "integrated_dataset.csv")
    integrated.to_csv(out_path, index=False)

    print(f"\n[Integration] integrated_dataset.csv")
    print(f"  Rows    : {len(integrated):,}")
    print(f"  Columns : {list(integrated.columns)}")
    print(f"  Target  : GHI_mean_J (J/m²/day)  ←  used for RMSE/MAE")
    print(f"  Target  : {target_col} (kWh/m²/day) ←  kept for reference")
    print(f"\n  Target statistics:")
    print(f"    kWh/m²/day : mean={integrated[target_col].mean():.4f}  "
          f"std={integrated[target_col].std():.4f}")
    print(f"    J/m²/day   : mean={integrated['GHI_mean_J'].mean():,.0f}  "
          f"std={integrated['GHI_mean_J'].std():,.0f}")
    print(f"\n  Saved → {out_path}")
    print("\n[Integration] Complete. Next → model_training.py")


if __name__ == "__main__":
    integrate_datasets()
