import os
import json
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    layout="wide",
    page_title="Climate Digital Twin Dashboard",
    initial_sidebar_state="expanded",
)

# =========================================================
# CUSTOM STYLING
# =========================================================

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #f7f9fc 0%, #eef3f9 100%);
    }

    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        color: #12344d;
        margin-bottom: 0.1rem;
    }

    .subtitle {
        font-size: 1rem;
        color: #4f6b81;
        margin-bottom: 1rem;
    }

    .section-card {
        background: white;
        padding: 1rem 1.2rem;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(16, 24, 40, 0.06);
        border: 1px solid rgba(18, 52, 77, 0.06);
        margin-bottom: 1rem;
    }

    .small-note {
        color: #5b6b79;
        font-size: 0.92rem;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid rgba(18, 52, 77, 0.06);
        padding: 12px 14px;
        border-radius: 14px;
        box-shadow: 0 6px 16px rgba(16, 24, 40, 0.05);
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 10px 14px;
        border: 1px solid rgba(18, 52, 77, 0.08);
    }

    .stTabs [aria-selected="true"] {
        background-color: #dff1ff !important;
        color: #12344d !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">Climate Digital Twin Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Regional hazard, storm-track, grid stress, curtailment, and postcode-level blackout risk for North East and Yorkshire.</div>',
    unsafe_allow_html=True,
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(__file__)

DATA_PATH = os.path.join(BASE_DIR, "data", "hazard_final.parquet")
GEO_PATH = os.path.join(BASE_DIR, "geo", "nuts3_NE_Yorkshire.geojson")

TRACK_CANDIDATES = [
    os.path.join(BASE_DIR, "data_raw", "C3S_StormTracks_ERA5_1979_2021_clean.csv"),
    os.path.join(BASE_DIR, "..", "data_raw", "C3S_StormTracks_ERA5_1979_2021_clean.csv"),
    os.path.join(BASE_DIR, "C3S_StormTracks_ERA5_1979_2021_clean.csv"),
]

CAPACITY_CANDIDATES = [
    os.path.join(BASE_DIR, "data_raw", "embedded-capacity-register-part-2.csv"),
    os.path.join(BASE_DIR, "..", "data_raw", "embedded-capacity-register-part-2.csv"),
]

CURTAIL_CANDIDATES = [
    os.path.join(BASE_DIR, "data_raw", "curtailment-events-site-specific.csv"),
    os.path.join(BASE_DIR, "..", "data_raw", "curtailment-events-site-specific.csv"),
]

FEEDER_CANDIDATES = [
    os.path.join(BASE_DIR, "data_raw", "npg-ehv-feeders.csv"),
    os.path.join(BASE_DIR, "..", "data_raw", "npg-ehv-feeders.csv"),
]

# =========================================================
# CONSTANTS
# =========================================================

REGION_LABELS = {
    "W_mean": "Wind Hazard Magnitude",
    "W_norm_year": "Regional Wind Severity Index",
    "W_sub_norm": "Local Wind Intensity Index",
    "MHI": "Multi-Hazard Impact Index",
    "P_fail": "Grid Failure Probability",
    "Curtailment_Risk": "Energy Curtailment Risk",
    "Node_Failure_Pressure": "Substation Stress Index",
    "P_fail_scenario": "Scenario Grid Failure Probability",
    "MHI_scenario": "Scenario Multi-Hazard Impact Index",
    "Curtailment_Risk_Scenario": "Scenario Curtailment Risk",
    "Node_Failure_Scenario": "Scenario Substation Stress Index",
}

NUTS_MAP = {
    "North East": [
        "Durham CC",
        "Northumberland",
        "Sunderland",
        "Tyneside",
        "Darlington",
        "Hartlepool and Stockton-on-Tees",
        "South Teesside",
    ],
    "Yorkshire and The Humber": [
        "Leeds",
        "Sheffield",
        "Bradford",
        "York",
        "Wakefield",
        "Calderdale and Kirklees",
        "Barnsley, Doncaster and Rotherham",
        "North Yorkshire CC",
        "East Riding of Yorkshire",
        "Kingston upon Hull, City of",
        "North and North East Lincolnshire",
    ],
}

SUB_TO_PARENT = {
    sub: parent
    for parent, subs in NUTS_MAP.items()
    for sub in subs
}

MAP_CENTER = {"lat": 54.5, "lon": -1.8}
DEFAULT_ZOOM = 5.0

# =========================================================
# HELPERS
# =========================================================

def find_existing_path(paths: list[str]) -> str | None:
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def read_csv_flexible(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path, low_memory=False, encoding="utf-8")
    except Exception:
        return pd.read_csv(path, low_memory=False, encoding="latin1")


def clean_string_series(series: pd.Series, fallback: str = "Unknown") -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace(
        {
            "": fallback,
            "nan": fallback,
            "None": fallback,
            "Data Not Available": fallback,
            "Data Not Applicable": fallback,
        }
    )
    return s.fillna(fallback)


def safe_normalise(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce").fillna(0.0)
    min_v = s.min()
    max_v = s.max()
    if max_v > min_v:
        return (s - min_v) / (max_v - min_v)
    return pd.Series(np.zeros(len(s)), index=s.index)


def sample_tracks_for_speed(df_tracks: pd.DataFrame, max_rows: int = 3500) -> pd.DataFrame:
    if df_tracks.empty or len(df_tracks) <= max_rows:
        return df_tracks

    n_years = max(1, df_tracks["year"].nunique())
    per_year = max(60, max_rows // n_years)

    sampled = (
        df_tracks.groupby("year", group_keys=False)
        .apply(lambda x: x.sample(min(len(x), per_year), random_state=42))
        .reset_index(drop=True)
    )

    if len(sampled) > max_rows:
        sampled = sampled.sample(max_rows, random_state=42).reset_index(drop=True)

    return sampled


def apply_map_layout(fig, height: int = 560):
    fig.update_layout(
        mapbox_style="carto-positron",
        height=height,
        margin=dict(l=0, r=0, t=10, b=0),
    )
    return fig


# =========================================================
# LOADERS - BASE DATA
# =========================================================

@st.cache_data(ttl=3600)
def load_parent_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path).copy()


@st.cache_data(ttl=3600)
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# =========================================================
# LOADERS - LIVE DATA
# =========================================================

@st.cache_data(ttl=3600)
def load_live_storms_noaa() -> pd.DataFrame:
    """
    Simple live storm proxy from NOAA global-hourly.
    This is not a full storm-track API, but gives live wind-related proxy data.
    """
    try:
        url = "https://www.ncei.noaa.gov/access/services/data/v1"
        params = {
            "dataset": "global-hourly",
            "stations": "01001099999",
            "startDate": "2023-01-01",
            "endDate": datetime.now().strftime("%Y-%m-%d"),
            "format": "json",
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return pd.DataFrame()

        data = response.json()
        df_live = pd.DataFrame(data)

        if df_live.empty:
            return df_live

        if "WND" in df_live.columns:
            wnd_parts = df_live["WND"].astype(str).str.split(",", expand=True)
            if wnd_parts.shape[1] >= 4:
                df_live["value"] = pd.to_numeric(wnd_parts[3], errors="coerce").fillna(1.0)
            else:
                df_live["value"] = 1.0
        else:
            df_live["value"] = 1.0

        if "DATE" not in df_live.columns:
            return pd.DataFrame()

        df_live["year"] = pd.to_datetime(df_live["DATE"], errors="coerce").dt.year
        df_live["storm_id"] = df_live.index.astype(str)

        # proxy location
        df_live["latitude"] = 54.5
        df_live["longitude"] = -1.8

        df_live = df_live.dropna(subset=["year"]).copy()
        df_live["year"] = df_live["year"].astype(int)

        return df_live[["latitude", "longitude", "year", "storm_id", "value"]]

    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600)
def load_live_grid_eso() -> pd.DataFrame:
    """
    Simple live grid proxy from National Grid ESO CKAN endpoint.
    """
    try:
        url = "https://api.nationalgrideso.com/api/3/action/datastore_search"
        params = {
            "resource_id": "d6a4bf54-c63f-4014-a716-49fd3878ca52",
            "limit": 1000,
        }

        response = requests.get(url, params=params, timeout=10)
        if response.status_code != 200:
            return pd.DataFrame()

        payload = response.json()
        records = payload.get("result", {}).get("records", [])
        df_live = pd.DataFrame(records)

        if df_live.empty:
            return df_live

        df_live["lat"] = 54.5
        df_live["lon"] = -1.8
        df_live["capacity_mw"] = pd.to_numeric(df_live.get("quantity", 0), errors="coerce").fillna(0.0)
        df_live["Postcode"] = "LIVE_GRID"
        df_live["Local Authority"] = "National Grid ESO"
        df_live["Energy Source 1"] = df_live.get("fuelType", "Unknown")

        return df_live[["lat", "lon", "capacity_mw", "Postcode", "Local Authority", "Energy Source 1"]]

    except Exception:
        return pd.DataFrame()


# =========================================================
# LOADERS - LOCAL DATA
# =========================================================

@st.cache_data(ttl=3600)
def load_tracks(paths: list[str]) -> pd.DataFrame:
    track_path = find_existing_path(paths)
    if track_path is None:
        return pd.DataFrame()

    tracks = read_csv_flexible(track_path).copy()

    rename_map = {}
    for c in tracks.columns:
        cl = str(c).lower().strip()

        if cl in ["lat", "latitude"]:
            rename_map[c] = "latitude"
        elif cl in ["lon", "longitude", "lng", "long"]:
            rename_map[c] = "longitude"
        elif cl == "year":
            rename_map[c] = "year"
        elif cl == "value":
            rename_map[c] = "value"
        elif cl in ["storm_id", "id", "track_id", "stormid"]:
            rename_map[c] = "storm_id"

    tracks = tracks.rename(columns=rename_map)

    required = {"latitude", "longitude", "year"}
    if not required.issubset(tracks.columns):
        return pd.DataFrame()

    tracks = tracks.dropna(subset=["latitude", "longitude", "year"]).copy()

    tracks["latitude"] = pd.to_numeric(tracks["latitude"], errors="coerce")
    tracks["longitude"] = pd.to_numeric(tracks["longitude"], errors="coerce")
    tracks["year"] = pd.to_numeric(tracks["year"], errors="coerce")

    tracks = tracks.dropna(subset=["latitude", "longitude", "year"]).copy()
    tracks["year"] = tracks["year"].astype(int)

    if "value" not in tracks.columns:
        tracks["value"] = 1.0
    else:
        tracks["value"] = pd.to_numeric(tracks["value"], errors="coerce").fillna(1.0)

    if "storm_id" not in tracks.columns:
        tracks["storm_id"] = tracks.groupby("year").cumcount().astype(str)
    else:
        tracks["storm_id"] = tracks["storm_id"].astype(str)

    tracks["longitude"] = tracks["longitude"].apply(lambda x: x - 360 if x > 180 else x)

    return tracks


@st.cache_data(ttl=3600)
def load_capacity(paths: list[str]) -> pd.DataFrame:
    path = find_existing_path(paths)
    if path is None:
        return pd.DataFrame()

    df_capacity = read_csv_flexible(path).copy()
    df_capacity.columns = df_capacity.columns.str.strip()

    # coordinates from geopoint
    if "geopoint" in df_capacity.columns:
        coords = df_capacity["geopoint"].astype(str).str.split(",", expand=True)
        if coords.shape[1] == 2:
            df_capacity["lat"] = pd.to_numeric(coords[0].str.strip(), errors="coerce")
            df_capacity["lon"] = pd.to_numeric(coords[1].str.strip(), errors="coerce")

    capacity_candidates = [
        "Energy Source & Energy Conversion Technology 1 - Registered Capacity (MW)",
        "Already connected Registered Capacity (MW)",
        "Already connected Registered Capacity (MW) ",
        "Maximum Export Capacity (MW)",
    ]

    for col in capacity_candidates:
        if col in df_capacity.columns:
            df_capacity["capacity_mw"] = pd.to_numeric(df_capacity[col], errors="coerce")
            break

    if "capacity_mw" not in df_capacity.columns:
        df_capacity["capacity_mw"] = np.nan

    if "Energy Source 1" not in df_capacity.columns:
        df_capacity["Energy Source 1"] = "Unknown"

    if "Postcode" not in df_capacity.columns:
        df_capacity["Postcode"] = "Unknown"

    if "Local Authority" not in df_capacity.columns:
        df_capacity["Local Authority"] = "Unknown"

    df_capacity["Postcode"] = clean_string_series(df_capacity["Postcode"], fallback="Unknown")
    df_capacity["Local Authority"] = clean_string_series(df_capacity["Local Authority"], fallback="Unknown")
    df_capacity["Energy Source 1"] = clean_string_series(df_capacity["Energy Source 1"], fallback="Unknown")

    df_capacity["capacity_mw"] = pd.to_numeric(df_capacity["capacity_mw"], errors="coerce").fillna(0.0)

    if "lat" in df_capacity.columns and "lon" in df_capacity.columns:
        df_capacity = df_capacity.dropna(subset=["lat", "lon"]).copy()
    else:
        return pd.DataFrame()

    return df_capacity


@st.cache_data(ttl=3600)
def load_curtailment(paths: list[str]) -> pd.DataFrame:
    path = find_existing_path(paths)
    if path is None:
        return pd.DataFrame()

    df_curtail = read_csv_flexible(path).copy()

    if "Start time UTC" in df_curtail.columns:
        df_curtail["start_ts"] = pd.to_datetime(df_curtail["Start time UTC"], errors="coerce")
        df_curtail["year"] = df_curtail["start_ts"].apply(lambda x: x.year if pd.notnull(x) else np.nan)
    else:
        df_curtail["year"] = np.nan

    energy_col = "Outage related curtailment-Total energy reduction (MWh)"
    avg_col = "Average access reduction (MW)"

    df_curtail["curtailment_mwh"] = pd.to_numeric(df_curtail.get(energy_col, np.nan), errors="coerce")
    df_curtail["avg_access_mw"] = pd.to_numeric(df_curtail.get(avg_col, np.nan), errors="coerce")

    if "Site" not in df_curtail.columns:
        df_curtail["Site"] = "Unknown"
    if "Reason For curtailment" not in df_curtail.columns:
        df_curtail["Reason For curtailment"] = "Unknown"
    if "Event ID" not in df_curtail.columns:
        df_curtail["Event ID"] = np.arange(len(df_curtail)).astype(str)

    df_curtail["Site"] = clean_string_series(df_curtail["Site"], fallback="Unknown")
    df_curtail["Reason For curtailment"] = clean_string_series(df_curtail["Reason For curtailment"], fallback="Unknown")

    return df_curtail


@st.cache_data(ttl=3600)
def load_feeders(paths: list[str]) -> pd.DataFrame:
    path = find_existing_path(paths)
    if path is None:
        return pd.DataFrame()

    df_feeders = read_csv_flexible(path).copy()

    if "Geo Point" in df_feeders.columns:
        coords = df_feeders["Geo Point"].astype(str).str.split(",", expand=True)
        if coords.shape[1] == 2:
            df_feeders["lat"] = pd.to_numeric(coords[0].str.strip(), errors="coerce")
            df_feeders["lon"] = pd.to_numeric(coords[1].str.strip(), errors="coerce")

    if "lat" not in df_feeders.columns or "lon" not in df_feeders.columns:
        return pd.DataFrame()

    df_feeders = df_feeders.dropna(subset=["lat", "lon"]).copy()

    if "Line situation" not in df_feeders.columns:
        df_feeders["Line situation"] = "Unknown"

    df_feeders["Line situation"] = clean_string_series(df_feeders["Line situation"], fallback="Unknown")
    df_feeders["voltage_numeric"] = pd.to_numeric(df_feeders.get("voltage", np.nan), errors="coerce").fillna(0.0)

    return df_feeders


# =========================================================
# REGIONAL MODEL
# =========================================================

@st.cache_data(ttl=3600)
def expand_to_subregions(df_parent_in: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in df_parent_in.iterrows():
        parent = r["region"]
        subs = NUTS_MAP.get(parent, [])
        if not subs:
            continue

        weights = np.linspace(0.90, 1.10, len(subs))

        for sub, w in zip(subs, weights):
            rows.append(
                {
                    "parent_region": parent,
                    "region": sub,
                    "year": int(r["year"]),
                    "W_mean": float(r["W_mean"]) * float(w),
                    "n_storm_pts": float(r.get("n_storm_pts", 0)),
                    "W_mean_norm_region": float(r.get("W_mean_norm_region", 0)) * float(w),
                }
            )

    df_regional = pd.DataFrame(rows)

    all_years = sorted(df_regional["year"].unique())
    all_regions = sorted(df_regional["region"].unique())

    full_index = pd.MultiIndex.from_product(
        [all_years, all_regions],
        names=["year", "region"],
    )

    df_regional = (
        df_regional.set_index(["year", "region"])
        .reindex(full_index)
        .reset_index()
    )

    df_regional["parent_region"] = df_regional["region"].map(SUB_TO_PARENT)
    df_regional["W_mean"] = df_regional["W_mean"].fillna(0.0)
    df_regional["n_storm_pts"] = df_regional["n_storm_pts"].fillna(0.0)
    df_regional["W_mean_norm_region"] = df_regional["W_mean_norm_region"].fillna(0.0)

    df_regional["W_norm_year"] = df_regional.groupby("year")["W_mean"].transform(
        lambda s: s / s.max() if s.max() > 0 else 0.0
    )

    df_regional["W_sub_norm"] = df_regional.groupby("region")["W_mean"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0
    )

    df_regional["MHI"] = 0.70 * df_regional["W_norm_year"] + 0.30 * df_regional["W_sub_norm"]

    alpha = 1.8
    xi = 0.5 * df_regional["W_sub_norm"] + 0.5 * df_regional["W_norm_year"]
    df_regional["P_fail"] = 1.0 - np.exp(-alpha * df_regional["MHI"] * xi)

    df_regional["Curtailment_Risk"] = np.clip(
        0.60 * df_regional["MHI"] + 0.40 * df_regional["P_fail"], 0, 1
    )

    df_regional["Node_Failure_Pressure"] = np.clip(
        0.55 * df_regional["P_fail"] + 0.45 * df_regional["W_sub_norm"], 0, 1
    )

    return df_regional


# =========================================================
# LOCAL GRID MODELS
# =========================================================

@st.cache_data(ttl=3600)
def build_local_grid_risk(
    capacity_df: pd.DataFrame,
    feeder_df: pd.DataFrame,
) -> pd.DataFrame:
    frames = []

    if not capacity_df.empty:
        cap = capacity_df.copy()
        cap["point_type"] = "Embedded Capacity Site"
        cap["label"] = cap["Postcode"]
        cap["risk_score"] = pd.to_numeric(cap["capacity_mw"], errors="coerce").fillna(0.0)
        cap["detail"] = cap["Energy Source 1"]
        frames.append(cap[["lat", "lon", "point_type", "label", "risk_score", "detail"]])

    if not feeder_df.empty:
        fd = feeder_df.copy()
        fd["point_type"] = "Grid Feeder"
        fd["label"] = fd["Line situation"]
        fd["risk_score"] = pd.to_numeric(fd["voltage_numeric"], errors="coerce").fillna(0.0)
        fd["detail"] = fd["Line situation"]
        frames.append(fd[["lat", "lon", "point_type", "label", "risk_score", "detail"]])

    if not frames:
        return pd.DataFrame()

    local = pd.concat(frames, ignore_index=True)
    local["risk_norm"] = safe_normalise(local["risk_score"])
    return local


@st.cache_data(ttl=3600)
def build_postcode_outage_model(capacity_df: pd.DataFrame, hazard_df: pd.DataFrame) -> pd.DataFrame:
    if capacity_df.empty:
        return pd.DataFrame()

    sites = capacity_df.copy().dropna(subset=["lat", "lon"])
    sites["capacity_norm"] = safe_normalise(sites["capacity_mw"])

    hazard_level = float(hazard_df["MHI"].mean()) if not hazard_df.empty else 0.0
    sites["grid_stress"] = 0.6 * sites["capacity_norm"] + 0.4 * hazard_level

    alpha = 1.6
    sites["outage_probability"] = 1 - np.exp(-alpha * sites["grid_stress"])
    sites["expected_curtailment_mw"] = sites["capacity_mw"] * sites["outage_probability"]

    return sites


@st.cache_data(ttl=3600)
def build_storm_blackout_model(
    capacity_df: pd.DataFrame,
    tracks_df: pd.DataFrame,
    hazard_df: pd.DataFrame,
) -> pd.DataFrame:
    if capacity_df.empty:
        return pd.DataFrame()

    sites = capacity_df.copy().dropna(subset=["lat", "lon"])
    sites["capacity_norm"] = safe_normalise(sites["capacity_mw"])

    storm_pressure = float(hazard_df["MHI"].mean()) if not hazard_df.empty else 0.0

    if not tracks_df.empty:
        storm_density = min(len(tracks_df) / 5000.0, 1.0)
    else:
        storm_density = 0.2

    sites["grid_stress_index"] = (
        0.5 * sites["capacity_norm"] +
        0.3 * storm_pressure +
        0.2 * storm_density
    )

    alpha = 1.8
    sites["blackout_probability"] = 1 - np.exp(-alpha * sites["grid_stress_index"])
    sites["expected_curtailment_mw"] = sites["capacity_mw"] * sites["blackout_probability"]

    return sites


# =========================================================
# SOURCE DATA
# =========================================================

df_parent = load_parent_data(DATA_PATH)
geojson = load_geojson(GEO_PATH)

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Dashboard Controls")

use_live = st.sidebar.toggle(
    "Live Digital Twin Mode",
    value=True,
    key="live_mode_toggle",
)

selected_parent_placeholder = st.sidebar.empty()
selected_year_placeholder = st.sidebar.empty()
scenario_placeholder = st.sidebar.empty()
show_regions_placeholder = st.sidebar.empty()

st.sidebar.markdown("---")
st.sidebar.caption("Live mode uses cached API calls to improve speed and stability.")

# =========================================================
# LOAD DATA ACCORDING TO MODE
# =========================================================

if use_live:
    tracks_live = load_live_storms_noaa()
    capacity_live = load_live_grid_eso()

    tracks = tracks_live if not tracks_live.empty else load_tracks(TRACK_CANDIDATES)
    capacity = capacity_live if not capacity_live.empty else load_capacity(CAPACITY_CANDIDATES)
else:
    tracks = load_tracks(TRACK_CANDIDATES)
    capacity = load_capacity(CAPACITY_CANDIDATES)

curtail = load_curtailment(CURTAIL_CANDIDATES)
feeders = load_feeders(FEEDER_CANDIDATES)

tracks = sample_tracks_for_speed(tracks, max_rows=3500)

# =========================================================
# PREP REGIONAL DATA
# =========================================================

df = expand_to_subregions(df_parent)

selected_parent = selected_parent_placeholder.selectbox(
    "Select Main Region",
    sorted(df["parent_region"].dropna().unique()),
)

selected_year = selected_year_placeholder.slider(
    "Select Year",
    int(df["year"].min()),
    int(df["year"].max()),
    int(df["year"].min()),
)

scenario = scenario_placeholder.selectbox(
    "Scenario",
    ["Baseline", "Mild", "Gradual", "Escalation"],
)

show_both_regions = show_regions_placeholder.checkbox(
    "Show both main regions on map",
    value=True,
)

scenario_factor = {
    "Baseline": 1.00,
    "Mild": 1.10,
    "Gradual": 1.25,
    "Escalation": 1.50,
}[scenario]

# sidebar diagnostics
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Status**")
st.sidebar.caption(f"Storm rows: {len(tracks)}")
st.sidebar.caption(f"Capacity rows: {len(capacity)}")
st.sidebar.caption(f"Feeder rows: {len(feeders)}")
st.sidebar.caption(f"Curtailment rows: {len(curtail)}")

if use_live:
    st.sidebar.success("Live mode enabled")
else:
    st.sidebar.info("Static local datasets enabled")

# =========================================================
# FILTERS
# =========================================================

df_selected_parent = df[df["parent_region"] == selected_parent].copy()
df_selected_year = df_selected_parent[df_selected_parent["year"] == selected_year].copy()

if show_both_regions:
    df_map_year = df[df["year"] == selected_year].copy()
else:
    df_map_year = df_selected_year.copy()

df_map_year["MHI_scenario"] = np.clip(df_map_year["MHI"] * scenario_factor, 0, 1.5)
df_map_year["P_fail_scenario"] = 1.0 - np.exp(
    -1.8 * df_map_year["MHI_scenario"] * (0.5 + 0.5 * df_map_year["W_sub_norm"])
)
df_map_year["Curtailment_Risk_Scenario"] = np.clip(
    0.60 * df_map_year["MHI_scenario"] + 0.40 * df_map_year["P_fail_scenario"], 0, 1
)
df_map_year["Node_Failure_Scenario"] = np.clip(
    0.55 * df_map_year["P_fail_scenario"] + 0.45 * df_map_year["W_sub_norm"], 0, 1
)

df_selected_parent["MHI_scenario"] = np.clip(df_selected_parent["MHI"] * scenario_factor, 0, 1.5)
df_selected_parent["P_fail_scenario"] = 1.0 - np.exp(
    -1.8 * df_selected_parent["MHI_scenario"] * (0.5 + 0.5 * df_selected_parent["W_sub_norm"])
)

local_grid = build_local_grid_risk(capacity, feeders)
postcode_risk = build_postcode_outage_model(capacity, df_selected_year)
postcode_blackout = build_storm_blackout_model(capacity, tracks, df_selected_year)

# =========================================================
# KPI ROW
# =========================================================

k1, k2, k3, k4 = st.columns(4)
k1.metric("Selected Region", selected_parent)
k2.metric("Selected Year", int(selected_year))
k3.metric("Average Hazard Index", f"{df_selected_year['MHI'].mean():.2f}")
k4.metric("Average Grid Failure Probability", f"{df_selected_year['P_fail'].mean():.2f}")

# =========================================================
# TABS
# =========================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(
    [
        "Storm Animation",
        "ERA5 Storm Tracks",
        "Hazard Timeline",
        "Climate Risk Dashboard",
        "Scenario Simulation",
        "Storm Intensity Surface",
        "Extreme Storm Return Period",
        "Grid Failure & Curtailment",
        "Local Grid Infrastructure",
        "Postcode Blackout Risk",
    ]
)

# =========================================================
# TAB 1 - STORM ANIMATION
# =========================================================

with tab1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Storm Animation Map")

    anim_df = df.copy()
    if not show_both_regions:
        anim_df = anim_df[anim_df["parent_region"] == selected_parent].copy()

    fig_anim = px.choropleth(
        anim_df,
        geojson=geojson,
        locations="region",
        featureidkey="properties.NUTS_NAME",
        color="W_norm_year",
        animation_frame="year",
        color_continuous_scale="Turbo",
        range_color=[0, 1],
        labels=REGION_LABELS,
        hover_data={
            "parent_region": True,
            "region": True,
            "year": True,
            "W_mean": ":.2f",
            "W_norm_year": ":.2f",
            "MHI": ":.2f",
            "P_fail": ":.2f",
        },
    )

    fig_anim.update_geos(fitbounds="locations", visible=False)
    fig_anim.update_traces(marker_line_width=0.8, marker_line_color="black")
    fig_anim.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))

    st.plotly_chart(fig_anim, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 2 - STORM TRACKS
# =========================================================

with tab2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("ERA5 Storm Track Explorer")

    if tracks.empty:
        st.warning("Storm-track data could not be loaded from live or local sources.")
    else:
        subtab1, subtab2, subtab3 = st.tabs(
            ["Track Points", "Trajectory Lines", "Storm Impact Overlay"]
        )

        with subtab1:
            fig_tracks = px.scatter_mapbox(
                tracks,
                lat="latitude",
                lon="longitude",
                animation_frame="year",
                color="value",
                size="value",
                size_max=10,
                zoom=4.7,
                center=MAP_CENTER,
                color_continuous_scale="Turbo",
                hover_data={
                    "storm_id": True,
                    "year": True,
                    "latitude": ":.2f",
                    "longitude": ":.2f",
                    "value": ":.2f",
                },
            )
            apply_map_layout(fig_tracks, height=540)
            st.plotly_chart(fig_tracks, use_container_width=True)

        with subtab2:
            line_year = st.slider(
                "Select year for trajectory lines",
                int(tracks["year"].min()),
                int(tracks["year"].max()),
                int(tracks["year"].min()),
                key="line_year_slider",
            )

            tracks_line = tracks[tracks["year"] == line_year].copy()

            fig_lines = px.line_mapbox(
                tracks_line.sort_values(["storm_id"]),
                lat="latitude",
                lon="longitude",
                color="storm_id",
                line_group="storm_id",
                zoom=4.8,
                center=MAP_CENTER,
                hover_data={
                    "storm_id": True,
                    "year": True,
                    "value": ":.2f",
                },
            )
            fig_lines.update_layout(showlegend=False)
            apply_map_layout(fig_lines, height=540)
            st.plotly_chart(fig_lines, use_container_width=True)

        with subtab3:
            overlay_year = st.slider(
                "Select year for storm impact overlay",
                int(tracks["year"].min()),
                int(tracks["year"].max()),
                min(selected_year, int(tracks["year"].max())),
                key="overlay_year_slider",
            )

            tracks_overlay = tracks[tracks["year"] == overlay_year].copy()
            risk_overlay = df[df["year"] == overlay_year].copy()

            fig_overlay = px.choropleth_mapbox(
                risk_overlay,
                geojson=geojson,
                locations="region",
                featureidkey="properties.NUTS_NAME",
                color="P_fail",
                color_continuous_scale="RdYlBu_r",
                range_color=[0, 1],
                opacity=0.42,
                zoom=4.8,
                center=MAP_CENTER,
                hover_data={
                    "region": True,
                    "parent_region": True,
                    "P_fail": ":.2f",
                    "MHI": ":.2f",
                },
            )

            fig_overlay.add_scattermapbox(
                lat=tracks_overlay["latitude"],
                lon=tracks_overlay["longitude"],
                mode="markers",
                marker=dict(size=4, opacity=0.75),
                text=tracks_overlay["storm_id"].astype(str),
                name="Storm track points",
            )
            apply_map_layout(fig_overlay, height=540)
            st.plotly_chart(fig_overlay, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 3 - HAZARD TIMELINE
# =========================================================

with tab3:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Regional Hazard Timeline")

    yearly = (
        df_selected_parent.groupby("year", as_index=False)
        .agg(
            wind_hazard=("W_mean", "mean"),
            multi_hazard_index=("MHI", "mean"),
            grid_failure_probability=("P_fail", "mean"),
            curtailment_risk=("Curtailment_Risk", "mean"),
        )
    )

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(x=yearly["year"], y=yearly["wind_hazard"], mode="lines+markers", name="Wind Hazard Magnitude"))
    fig_timeline.add_trace(go.Scatter(x=yearly["year"], y=yearly["multi_hazard_index"], mode="lines+markers", name="Multi-Hazard Impact Index", yaxis="y2"))
    fig_timeline.add_trace(go.Scatter(x=yearly["year"], y=yearly["grid_failure_probability"], mode="lines+markers", name="Grid Failure Probability", yaxis="y2"))
    fig_timeline.add_trace(go.Scatter(x=yearly["year"], y=yearly["curtailment_risk"], mode="lines+markers", name="Energy Curtailment Risk", yaxis="y2"))

    fig_timeline.update_layout(
        height=520,
        xaxis_title="Year",
        yaxis=dict(title="Absolute Hazard Magnitude"),
        yaxis2=dict(title="Relative Risk Index", overlaying="y", side="right", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig_timeline, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 4 - CLIMATE RISK DASHBOARD
# =========================================================

with tab4:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Climate Risk Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected Year", selected_year)
    c2.metric("Average Wind Hazard", f"{df_selected_year['W_mean'].mean():.2f}")
    c3.metric("Average Multi-Hazard Index", f"{df_selected_year['MHI'].mean():.2f}")
    c4.metric("Average Grid Failure Probability", f"{df_selected_year['P_fail'].mean():.2f}")

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown("#### Scenario Grid Failure Probability Map")

        fig_risk = px.choropleth(
            df_map_year,
            geojson=geojson,
            locations="region",
            featureidkey="properties.NUTS_NAME",
            color="P_fail_scenario",
            color_continuous_scale="RdYlBu_r",
            range_color=[0, 1],
            labels=REGION_LABELS,
            hover_data={
                "parent_region": True,
                "region": True,
                "year": True,
                "MHI_scenario": ":.2f",
                "P_fail_scenario": ":.2f",
                "W_mean": ":.2f",
            },
        )
        fig_risk.update_geos(fitbounds="locations", visible=False)
        fig_risk.update_traces(marker_line_width=0.8, marker_line_color="black")
        fig_risk.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_risk, use_container_width=True)

    with right:
        st.markdown("#### Subregional Impact Ranking")

        mhi_rank = (
            df_selected_year[["region", "MHI", "P_fail"]]
            .sort_values("MHI", ascending=False)
        )

        fig_mhi_bar = px.bar(
            mhi_rank,
            x="MHI",
            y="region",
            orientation="h",
            color="P_fail",
            labels=REGION_LABELS,
            color_continuous_scale="Turbo",
            range_color=[0, 1],
        )
        fig_mhi_bar.update_layout(
            height=520,
            yaxis_title="Subregion",
            xaxis_title="Multi-Hazard Impact Index",
            margin=dict(l=0, r=0, t=10, b=0),
        )
        st.plotly_chart(fig_mhi_bar, use_container_width=True)

    st.markdown("#### Relative Hazard Heatmap")

    heat = df.pivot_table(
        index="region",
        columns="year",
        values="W_sub_norm",
        aggfunc="mean",
    )

    fig_heat = px.imshow(
        heat,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        zmin=0,
        zmax=1,
        labels={"color": "Local Wind Intensity Index"},
    )
    fig_heat.update_layout(
        height=440,
        coloraxis_colorbar_title="Relative Severity",
        xaxis_title="Year",
        yaxis_title="Subregion",
    )
    st.plotly_chart(fig_heat, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 5 - SCENARIO SIMULATION
# =========================================================

with tab5:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Scenario Simulation")

    scenario_df = (
        df_selected_parent.groupby("year", as_index=False)
        .agg(
            MHI=("MHI", "mean"),
            P_fail=("P_fail", "mean"),
            Curtailment_Risk=("Curtailment_Risk", "mean"),
        )
    )

    scen_list = []
    for name, fac, stress in [
        ("Mild", 1.10, 0.75),
        ("Gradual", 1.25, 0.85),
        ("Escalation", 1.50, 0.95),
    ]:
        temp = scenario_df.copy()
        temp["Scenario"] = name
        temp["MHI_scenario"] = np.clip(temp["MHI"] * fac, 0, 1.5)
        temp["P_fail_scenario"] = 1.0 - np.exp(-1.8 * temp["MHI_scenario"] * stress)
        temp["Curtailment_scenario"] = np.clip(
            0.60 * temp["MHI_scenario"] + 0.40 * temp["P_fail_scenario"], 0, 1
        )
        scen_list.append(temp)

    scen_all = pd.concat(scen_list, ignore_index=True)

    col_a, col_b = st.columns(2)

    with col_a:
        fig_scen_mhi = px.line(
            scen_all,
            x="year",
            y="MHI_scenario",
            color="Scenario",
            markers=True,
        )
        fig_scen_mhi.update_layout(height=420, xaxis_title="Year", yaxis_title="Scenario Multi-Hazard Index")
        st.plotly_chart(fig_scen_mhi, use_container_width=True)

    with col_b:
        fig_scen_pf = px.line(
            scen_all,
            x="year",
            y="P_fail_scenario",
            color="Scenario",
            markers=True,
        )
        fig_scen_pf.update_layout(height=420, xaxis_title="Year", yaxis_title="Scenario Grid Failure Probability")
        st.plotly_chart(fig_scen_pf, use_container_width=True)

    fig_scen_curt = px.line(
        scen_all,
        x="year",
        y="Curtailment_scenario",
        color="Scenario",
        markers=True,
    )
    fig_scen_curt.update_layout(height=380, xaxis_title="Year", yaxis_title="Scenario Curtailment Risk")
    st.plotly_chart(fig_scen_curt, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 6 - STORM INTENSITY SURFACE
# =========================================================

with tab6:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Storm Intensity Surface")

    if tracks.empty:
        st.warning("Storm-track data could not be loaded.")
    else:
        surface_year = st.slider(
            "Select year for storm intensity surface",
            int(tracks["year"].min()),
            int(tracks["year"].max()),
            int(tracks["year"].min()),
            key="surface_year_slider",
        )

        tracks_surface = tracks[tracks["year"] == surface_year].copy()

        fig_surface = px.density_mapbox(
            tracks_surface,
            lat="latitude",
            lon="longitude",
            z="value",
            radius=18,
            center=MAP_CENTER,
            zoom=4.8,
            color_continuous_scale="Turbo",
        )
        apply_map_layout(fig_surface, height=540)
        st.plotly_chart(fig_surface, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 7 - RETURN PERIOD
# =========================================================

with tab7:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Extreme Storm Return Period Analysis")
    st.caption("Empirical extreme-value style approximation using annual maximum hazard.")

    rp_df = (
        df_selected_parent.groupby("year", as_index=False)
        .agg(annual_max_hazard=("W_mean", "max"))
        .sort_values("annual_max_hazard", ascending=False)
        .reset_index(drop=True)
    )

    n = len(rp_df)
    rp_df["rank"] = np.arange(1, n + 1)
    rp_df["exceedance_probability"] = rp_df["rank"] / (n + 1)
    rp_df["return_period_years"] = 1.0 / rp_df["exceedance_probability"]

    col1, col2 = st.columns(2)

    with col1:
        fig_rp = px.scatter(
            rp_df,
            x="return_period_years",
            y="annual_max_hazard",
            hover_data={"year": True},
            log_x=True,
        )
        fig_rp.update_layout(
            height=420,
            xaxis_title="Return Period (Years, log scale)",
            yaxis_title="Annual Maximum Hazard",
        )
        st.plotly_chart(fig_rp, use_container_width=True)

    with col2:
        fig_rank = px.bar(
            rp_df.sort_values("year"),
            x="year",
            y="annual_max_hazard",
        )
        fig_rank.update_layout(
            height=420,
            xaxis_title="Year",
            yaxis_title="Annual Maximum Hazard",
        )
        st.plotly_chart(fig_rank, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 8 - GRID FAILURE & CURTAILMENT
# =========================================================

with tab8:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Grid Failure and Curtailment Dashboard")

    current_parent = df_selected_year.copy()
    left, right = st.columns(2)

    with left:
        st.markdown("#### Substation Stress Map")

        fig_node = px.choropleth(
            current_parent,
            geojson=geojson,
            locations="region",
            featureidkey="properties.NUTS_NAME",
            color="Node_Failure_Pressure",
            color_continuous_scale="Reds",
            range_color=[0, 1],
            hover_data={
                "region": True,
                "Node_Failure_Pressure": ":.2f",
                "P_fail": ":.2f",
                "MHI": ":.2f",
            },
        )
        fig_node.update_geos(fitbounds="locations", visible=False)
        fig_node.update_traces(marker_line_width=0.8, marker_line_color="black")
        fig_node.update_layout(height=520, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_node, use_container_width=True)

    with right:
        st.markdown("#### Energy Curtailment Risk Ranking")

        fig_curt = px.bar(
            current_parent.sort_values("Curtailment_Risk", ascending=False),
            x="Curtailment_Risk",
            y="region",
            orientation="h",
            color="Curtailment_Risk",
            color_continuous_scale="OrRd",
            range_color=[0, 1],
        )
        fig_curt.update_layout(
            height=520,
            xaxis_title="Energy Curtailment Risk",
            yaxis_title="Subregion",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_curt, use_container_width=True)

    summary = current_parent[
        ["region", "Node_Failure_Pressure", "Curtailment_Risk", "P_fail", "MHI"]
    ].sort_values("Curtailment_Risk", ascending=False)

    summary = summary.rename(
        columns={
            "region": "Subregion",
            "Node_Failure_Pressure": "Substation Stress Index",
            "Curtailment_Risk": "Energy Curtailment Risk",
            "P_fail": "Grid Failure Probability",
            "MHI": "Multi-Hazard Impact Index",
        }
    )

    st.markdown("#### Subregional Summary")
    st.dataframe(summary, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 9 - LOCAL GRID INFRASTRUCTURE
# =========================================================

with tab9:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Local Grid Infrastructure and Site-Specific Risk")

    c1, c2, c3 = st.columns(3)
    c1.metric("Embedded Generation Sites", len(capacity))
    c2.metric("Curtailment Events", len(curtail))
    c3.metric("Grid Feeders", len(feeders))

    subtab1, subtab2, subtab3, subtab4 = st.tabs(
        ["Embedded Capacity", "Feeder Network", "Curtailment Timeline", "Local Risk Overlay"]
    )

    with subtab1:
        st.markdown("#### Embedded Generation Sites by Postcode")

        if capacity.empty or capacity["lat"].dropna().empty:
            st.info("Embedded capacity dataset was found, but no usable coordinate records were parsed.")
        else:
            fig_cap = px.scatter_mapbox(
                capacity.dropna(subset=["lat", "lon"]),
                lat="lat",
                lon="lon",
                color="Energy Source 1",
                size="capacity_mw",
                hover_data={
                    "Postcode": True,
                    "Local Authority": True,
                    "capacity_mw": ":.2f",
                    "Energy Source 1": True,
                },
                zoom=5,
                center=MAP_CENTER,
            )
            apply_map_layout(fig_cap, height=520)
            st.plotly_chart(fig_cap, use_container_width=True)

            cap_preview = (
                capacity[["Postcode", "Local Authority", "Energy Source 1", "capacity_mw"]]
                .sort_values("capacity_mw", ascending=False)
                .head(20)
            )
            st.dataframe(cap_preview, use_container_width=True)

    with subtab2:
        st.markdown("#### Grid Feeder Network")

        if feeders.empty:
            st.info("Feeder dataset not available.")
        else:
            fig_feed = px.scatter_mapbox(
                feeders,
                lat="lat",
                lon="lon",
                color="Line situation",
                size="voltage_numeric",
                hover_data={
                    "Line situation": True,
                    "voltage_numeric": ":.0f",
                },
                zoom=5,
                center=MAP_CENTER,
            )
            apply_map_layout(fig_feed, height=520)
            st.plotly_chart(fig_feed, use_container_width=True)

    with subtab3:
        st.markdown("#### Curtailment Timeline and Site Hotspots")

        if curtail.empty:
            st.info("Curtailment dataset not available.")
        else:
            cur_year = (
                curtail.groupby("year", as_index=False)
                .agg(
                    events=("Event ID", "count"),
                    curtailment_mwh=("curtailment_mwh", "sum"),
                )
                .dropna(subset=["year"])
            )

            col_a, col_b = st.columns(2)

            with col_a:
                fig_cur_events = px.line(cur_year, x="year", y="events", markers=True)
                fig_cur_events.update_layout(height=360, xaxis_title="Year", yaxis_title="Curtailment Events")
                st.plotly_chart(fig_cur_events, use_container_width=True)

            with col_b:
                fig_cur_mwh = px.line(cur_year, x="year", y="curtailment_mwh", markers=True)
                fig_cur_mwh.update_layout(height=360, xaxis_title="Year", yaxis_title="Curtailment Energy (MWh)")
                st.plotly_chart(fig_cur_mwh, use_container_width=True)

            site_rank = (
                curtail.groupby("Site", as_index=False)
                .agg(
                    events=("Event ID", "count"),
                    curtailment_mwh=("curtailment_mwh", "sum"),
                    avg_access_mw=("avg_access_mw", "mean"),
                )
                .sort_values("curtailment_mwh", ascending=False)
                .head(20)
            )

            fig_site = px.bar(
                site_rank,
                x="curtailment_mwh",
                y="Site",
                orientation="h",
                color="events",
            )
            fig_site.update_layout(
                height=500,
                xaxis_title="Total Curtailed Energy (MWh)",
                yaxis_title="Site",
            )
            st.plotly_chart(fig_site, use_container_width=True)

    with subtab4:
        st.markdown("#### Future Local Outage and Curtailment Risk")

        if local_grid.empty:
            st.info("No local capacity or feeder points are available for overlay.")
        else:
            local = local_grid.copy()

            year_scale = (selected_year - df["year"].min()) / max(1, (df["year"].max() - df["year"].min()))
            local["future_risk"] = np.clip(
                0.45 * local["risk_norm"] + 0.35 * scenario_factor / 1.5 + 0.20 * year_scale,
                0,
                1,
            )

            fig_local = px.scatter_mapbox(
                local,
                lat="lat",
                lon="lon",
                color="future_risk",
                size="risk_norm",
                hover_data={
                    "point_type": True,
                    "label": True,
                    "detail": True,
                    "future_risk": ":.2f",
                },
                color_continuous_scale="Turbo",
                zoom=5,
                center=MAP_CENTER,
            )
            apply_map_layout(fig_local, height=540)
            st.plotly_chart(fig_local, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================================
# TAB 10 - POSTCODE BLACKOUT RISK
# =========================================================

with tab10:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Storm-Driven Postcode Blackout Risk Simulation")

    if postcode_blackout.empty:
        st.info("No postcode-level generation site data are available.")
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Generation Sites", len(postcode_blackout))
        c2.metric("Average Blackout Probability", f"{postcode_blackout['blackout_probability'].mean():.2f}")
        c3.metric("Expected Curtailment (MW)", f"{postcode_blackout['expected_curtailment_mw'].sum():.1f}")

        st.markdown("#### Postcode-Level Grid Failure Risk Map")

        fig_blackout = px.scatter_mapbox(
            postcode_blackout,
            lat="lat",
            lon="lon",
            size="capacity_mw",
            color="blackout_probability",
            hover_data={
                "Postcode": True,
                "Local Authority": True,
                "capacity_mw": ":.2f",
                "blackout_probability": ":.2f",
                "expected_curtailment_mw": ":.2f",
            },
            color_continuous_scale="Turbo",
            zoom=5,
            center=MAP_CENTER,
        )
        apply_map_layout(fig_blackout, height=560)
        st.plotly_chart(fig_blackout, use_container_width=True)

        st.markdown("#### Highest Blackout Risk Postcodes")

        risk_rank = (
            postcode_blackout[
                ["Postcode", "Local Authority", "capacity_mw", "blackout_probability", "expected_curtailment_mw"]
            ]
            .sort_values("blackout_probability", ascending=False)
            .head(20)
            .rename(
                columns={
                    "capacity_mw": "Connected Capacity (MW)",
                    "blackout_probability": "Blackout Probability",
                    "expected_curtailment_mw": "Expected Curtailment (MW)",
                }
            )
        )

        st.dataframe(risk_rank, use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)
