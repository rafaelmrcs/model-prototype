"""
data_processing.py
────────────────────────────────────────────────────────────────────────────
Cleans raw data files produced by data_acquisition.py.

Changes from original:
  + process_baseline_spatial() uncommented and activated
    — processes the 3,000-row spatial CSV that is the primary dataset
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd

ROOT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR       = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DIR = os.path.join(ROOT_DIR, "data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)


# =============================================================================
# A) Baseline spatial dataset — 3,000 rows: lat, lon, GHI_mean
# =============================================================================

def process_baseline_spatial(year: str = "2024") -> pd.DataFrame:
    """
    §2.1.3.4 Data Preprocessing — baseline spatial dataset.
    Cleans the 3,000-row spatial CSV produced by fetch_nasa_baseline_spatial().
    Steps per §2.1.3.4:
      - Cast lat, lon, GHI_mean to numeric
      - Drop rows with NaN in any critical column (removal of invalid records)
      - Clip GHI to physically plausible range [0.5, 10] kWh/m²/day
        (standardisation / removal of NASA fill-value artefacts)
    """
    city_slug = "davao_city"
    path = os.path.join(RAW_DIR,
                        f"baseline_spatial_dataset_{city_slug}_{year}.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing: {path}\n"
            "Run data_acquisition.fetch_nasa_baseline_spatial() first."
        )

    df        = pd.read_csv(path)
    target_col = f"GHI_mean_{year}"

    df["lat"]      = pd.to_numeric(df["lat"],      errors="coerce")
    df["lon"]      = pd.to_numeric(df["lon"],      errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    n_before = len(df)
    df = df.dropna(subset=["lat", "lon", target_col]).reset_index(drop=True)
    df[target_col] = df[target_col].clip(0.5, 10.0)

    print(f"[Process Baseline Spatial] {n_before} → {len(df)} rows "
          f"({n_before - len(df)} dropped)")
    print(f"  GHI_mean range: [{df[target_col].min():.3f}, "
          f"{df[target_col].max():.3f}] kWh/m²/day")
    return df


# =============================================================================
# B) NASA daily time-series (centroid) — unchanged from original
# =============================================================================

def process_nasa() -> pd.DataFrame:
    """
    §2.1.3.4 Data Preprocessing — centroid daily NASA time-series (optional).
    Replaces NASA fill value (-999) with NaN and applies linear interpolation
    to fill gaps, per §2.1.3.4 ("missing or incomplete data addressed using
    interpolation"). Not used in the main spatial pipeline.
    """
    path = os.path.join(RAW_DIR, "nasa_raw.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    df.replace(-999, np.nan, inplace=True)
    df.interpolate(method="linear", inplace=True)
    return df


# =============================================================================
# C) OSM buildings — unchanged from original
# =============================================================================

def process_osm() -> gpd.GeoDataFrame:
    """
    §2.1.3.4 Data Preprocessing + §2.2.3.2 Rooftop Area (Table 4).
    Loads and cleans OSM building footprints from osm_buildings.geojson.
    Reprojects to UTM Zone 51N (EPSG:32651) so geometry.area returns m²,
    then computes rooftop_area_sq_m — the first topographical feature (Table 4).
    Caps at 10,000 buildings to keep feature engineering tractable.
    """
    path = os.path.join(RAW_DIR, "osm_buildings.geojson")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}")

    gdf = gpd.read_file(path)
    gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    gdf = gdf.to_crs("EPSG:32651")           # metres — required for area calc
    gdf["rooftop_area_sq_m"] = gdf.geometry.area

    if len(gdf) > 10_000:
        gdf = gdf.sample(10_000, random_state=42).reset_index(drop=True)

    print(f"[Process OSM] {len(gdf):,} buildings  "
          f"area range: [{gdf['rooftop_area_sq_m'].min():.1f}, "
          f"{gdf['rooftop_area_sq_m'].max():.1f}] m²")
    return gdf


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("Processing data …")

    # PRIMARY dataset: 3,000 spatial points
    baseline_df = process_baseline_spatial(year="2024")
    baseline_df.to_csv(
        os.path.join(PROCESSED_DIR, "baseline_spatial_clean_2024.csv"),
        index=False,
    )
    print(f"  Saved baseline_spatial_clean_2024.csv  ({len(baseline_df)} rows)")

    # OSM buildings
    osm_gdf = process_osm()
    osm_gdf.to_file(
        os.path.join(PROCESSED_DIR, "osm_clean.geojson"),
        driver="GeoJSON",
    )
    print(f"  Saved osm_clean.geojson  ({len(osm_gdf):,} buildings)")

    # Daily time-series (optional — only needed if doing temporal analysis)
    # nasa_df = process_nasa()
    # nasa_df.to_csv(os.path.join(PROCESSED_DIR, "nasa_clean.csv"), index=False)

    print("Data processing complete.")
