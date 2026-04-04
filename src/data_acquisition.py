import os
import time
import requests
import numpy as np
import pandas as pd
import osmnx as ox
import geopandas as gpd
from shapely.geometry import Point

# --- Directories ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR  = os.path.join(ROOT_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

NASA_BASE_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# =============================================================================
# Helpers
# =============================================================================

def get_city_boundary(place_name: str) -> gpd.GeoDataFrame:
    """
    §2.1 Data Acquisition — helper to obtain the city polygon from OSM.
    Used by both fetch_nasa_baseline_spatial and fetch_osm_data to
    constrain sampling and data extraction to Davao City boundaries.
    """
    gdf = ox.geocode_to_gdf(place_name)
    if gdf.empty:
        raise ValueError(f"Could not geocode place boundary for: {place_name}")
    gdf = gdf.set_crs("EPSG:4326", allow_override=True)
    return gdf


def sample_random_points_in_polygon(polygon, n_points: int = 3000,
                                    seed: int = 42) -> gpd.GeoDataFrame:
    """
    §2.1.1 NASA Meteorological Data — spatial sampling strategy.
    Generates n_points uniformly distributed random coordinates inside the
    city boundary polygon. Replicates the Quezon City study methodology:
    "A Python script generated 3,000 random latitude and longitude coordinates
    within Quezon City's boundaries" (Sales et al., 2024 [16]).
    Uniform spatial distribution avoids clustering and ensures even city-wide
    coverage — same approach as the baseline reference study.
    """
    rng  = np.random.default_rng(seed)
    minx, miny, maxx, maxy = polygon.bounds
    pts  = []
    while len(pts) < n_points:
        lon = rng.uniform(minx, maxx)
        lat = rng.uniform(miny, maxy)
        p   = Point(lon, lat)
        if polygon.contains(p):
            pts.append(p)

    gdf       = gpd.GeoDataFrame(geometry=pts, crs="EPSG:4326")
    gdf["lon"] = gdf.geometry.x
    gdf["lat"] = gdf.geometry.y
    return gdf[["lat", "lon", "geometry"]]


def nasa_power_request(lat: float, lon: float,
                       start: str, end: str,
                       parameters: str) -> dict:
    """
    §2.1.1 NASA Meteorological Data — API query helper.
    Queries the NASA POWER public API (Table 3) for a single coordinate.
    Parameters fetched: ALLSKY_SFC_SW_DWN (GHI), T2M, RH2M, ALLSKY_KT.
    Community "RE" = Renewable Energy, format JSON.
    """
    params = {
        "parameters": parameters,
        "community":  "RE",
        "latitude":   lat,
        "longitude":  lon,
        "start":      start,
        "end":        end,
        "format":     "JSON",
    }
    r = requests.get(NASA_BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def extract_series(js: dict, param: str) -> pd.Series:
    """
    §2.1.1 NASA Meteorological Data — response parser helper.
    Extracts a single parameter time series from the NASA POWER JSON response
    and coerces it to numeric, replacing any non-numeric fill values with NaN.
    """
    series = js["properties"]["parameter"][param]
    return pd.to_numeric(pd.Series(series), errors="coerce")


# =============================================================================
# A) Baseline spatial dataset — replicates Quezon City methodology on Davao
#
# Quezon City study:
#   "A Python script generated 3,000 random latitude and longitude
#    coordinates within Quezon City's boundaries ... Utilizing PVWatts V8
#    API of NREL ... the optimal tilt solar irradiance for each generated
#    coordinate was calculated."
#
# This function does the equivalent for Davao City using NASA POWER
# (ALLSKY_SFC_SW_DWN = GHI, the equivalent of PVWatts solrad_annual).
#
# Output CSV schema: lat | lon | GHI_mean_{year}
#   Identical to the Quezon City baseline CSV: lat | lon | solarrad
# =============================================================================

def fetch_nasa_baseline_spatial(
        place_name: str  = "Davao City, Philippines",
        year:       str  = "2024",
        n_points:   int  = 3000,
        seed:       int  = 42,
        sleep_s:    float = 0.2,
) -> None:
    """
    §2.1.1 NASA Meteorological Data + §2.1 Data Acquisition (Table 2).
    Samples n_points random coordinates inside the city boundary and fetches
    the annual average GHI (ALLSKY_SFC_SW_DWN) from NASA POWER for each.
    This produces the PRIMARY dataset used for model training (3,000 rows),
    replicating the Quezon City study's spatial sampling methodology [16].
    Output schema: lat | lon | GHI_mean_{year}  (kWh/m²/day)
    Implements checkpoint/resume — saves progress every 50 rows so the
    long API fetch (15–30 min) can be safely interrupted and restarted.
    """
    print(f"[Baseline Spatial] Sampling {n_points} points in {place_name} …")

    boundary = get_city_boundary(place_name)
    poly     = boundary.geometry.iloc[0]
    pts      = sample_random_points_in_polygon(poly, n_points=n_points, seed=seed)

    city_slug = place_name.split(",")[0].lower().replace(" ", "_")
    out_path  = os.path.join(RAW_DIR,
                             f"baseline_spatial_dataset_{city_slug}_{year}.csv")
    tmp_path  = out_path.replace(".csv", "_partial.csv")

    # Resume from checkpoint if it exists
    if os.path.exists(tmp_path):
        done_df  = pd.read_csv(tmp_path)
        done_set = set(zip(done_df["lat"].round(6), done_df["lon"].round(6)))
        print(f"[Baseline Spatial] Resuming: {len(done_df)} rows done.")
    else:
        done_df  = pd.DataFrame(columns=["lat", "lon", f"GHI_mean_{year}"])
        done_set = set()

    start  = f"{year}0101"
    end    = f"{year}1231"
    buffer = []

    for _, row in pts.iterrows():
        lat = float(row["lat"])
        lon = float(row["lon"])
        key = (round(lat, 6), round(lon, 6))
        if key in done_set:
            continue

        try:
            js       = nasa_power_request(lat, lon, start, end,
                                          parameters="ALLSKY_SFC_SW_DWN")
            ghi      = extract_series(js, "ALLSKY_SFC_SW_DWN")
            ghi_mean = float(ghi.mean(skipna=True))
            buffer.append({"lat": lat, "lon": lon,
                           f"GHI_mean_{year}": ghi_mean})
        except Exception as e:
            print(f"  Error at ({lat:.4f}, {lon:.4f}): {e}")
            buffer.append({"lat": lat, "lon": lon,
                           f"GHI_mean_{year}": np.nan})

        # Checkpoint every 50 rows
        if len(buffer) >= 50:
            done_df = pd.concat([done_df, pd.DataFrame(buffer)],
                                ignore_index=True)
            done_df.to_csv(tmp_path, index=False)
            buffer.clear()
            print(f"[Baseline Spatial] Progress: {len(done_df)}/{n_points}")
        time.sleep(sleep_s)

    if buffer:
        done_df = pd.concat([done_df, pd.DataFrame(buffer)], ignore_index=True)

    done_df.to_csv(out_path, index=False)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    print(f"[Baseline Spatial] Done → {out_path}  ({len(done_df)} rows)")


# =============================================================================
# B) Time-series data (centroid) — kept for reference / other analyses
# =============================================================================

def fetch_nasa_timeseries(place_name: str = "Davao City, Philippines",
                          year:       str = "2024") -> None:
    """
    §2.1.1 NASA Meteorological Data — centroid daily time-series (optional).
    Fetches 365 daily records for the city centroid: GHI, T2M, RH2M, ALLSKY_KT.
    NOT used in the main spatial pipeline — kept for reference or supplementary
    temporal analysis. The primary pipeline uses fetch_nasa_baseline_spatial().
    """
    print(f"[Time Series] Fetching NASA POWER daily data for {place_name} …")

    boundary = get_city_boundary(place_name)
    poly     = boundary.geometry.iloc[0]
    centroid = poly.centroid
    lon      = float(centroid.x)
    lat      = float(centroid.y)

    start = f"{year}0101"
    end   = f"{year}1231"

    js       = nasa_power_request(lat, lon, start, end,
                                  parameters="ALLSKY_SFC_SW_DWN,T2M,RH2M,ALLSKY_KT")
    features = js["properties"]["parameter"]
    dates    = list(features["ALLSKY_SFC_SW_DWN"].keys())

    df = pd.DataFrame({
        "date":              dates,
        "ALLSKY_SFC_SW_DWN": list(features["ALLSKY_SFC_SW_DWN"].values()),
        "T2M":               list(features["T2M"].values()),
        "RH2M":              list(features["RH2M"].values()),
        "ALLSKY_KT":         list(features["ALLSKY_KT"].values()),
        "lat":               lat,
        "lon":               lon,
    })

    out_path = os.path.join(RAW_DIR, "nasa_raw.csv")
    df.to_csv(out_path, index=False)
    print(f"[Time Series] Saved → {out_path}  (lat={lat:.5f}, lon={lon:.5f})")


# =============================================================================
# C) OSM Buildings
# =============================================================================

def fetch_osm_data(place_name: str = "Davao City, Philippines") -> None:
    """
    §2.1.2 Topographical Data (Table 2, Table 4).
    Fetches building footprints from OpenStreetMap using osmnx.
    Davao City has >81,000 mapped structures; this extracts all Polygon and
    MultiPolygon building geometries (the shapes used to compute rooftop area,
    orientation, and shading in feature_engineering.py).
    Output: osm_buildings.geojson — one row per building polygon.
    """
    print(f"[OSM] Fetching building footprints for {place_name} …")
    try:
        gdf = ox.features_from_place(place_name, tags={"building": True})
        gdf = gdf[gdf.geometry.type.isin(["Polygon", "MultiPolygon"])]
        gdf = gdf[["geometry"]]
        out = os.path.join(RAW_DIR, "osm_buildings.geojson")
        gdf.to_file(out, driver="GeoJSON")
        print(f"[OSM] Saved → {out}  ({len(gdf):,} buildings)")
    except Exception as e:
        print(f"[OSM] Error: {e}")


# =============================================================================
# MAIN — run order matches Quezon City study methodology
# =============================================================================

if __name__ == "__main__":
    # 1) Baseline spatial dataset — 3,000 random coordinates → annual GHI
    #    This is the PRIMARY dataset for model training/testing.
    #    Replicates Quezon City study exactly (they used 3,000 coords + PVWatts).
    fetch_nasa_baseline_spatial(
        place_name="Davao City, Philippines",
        year="2024",
        n_points=3000,
        seed=42,
    )

    # 2) OSM building footprints — used to attach building features to each point
    fetch_osm_data()

    # 3) (Optional) daily time-series — kept for reference
    # fetch_nasa_timeseries()
