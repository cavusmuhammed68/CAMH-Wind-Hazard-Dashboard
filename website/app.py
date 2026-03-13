import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =================================================
# PAGE CONFIG
# =================================================

st.set_page_config(layout="wide", page_title="Climate Digital Twin Dashboard")

st.title("Climate Digital Twin Dashboard")
st.markdown("Regional hazard, storm-track, risk, climate digital twin, and local grid-risk dashboard")

# =================================================
# PATHS
# =================================================

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

# =================================================
# LOADERS
# =================================================

@st.cache_data
def load_parent_data(path: str) -> pd.DataFrame:
    return pd.read_parquet(path).copy()

@st.cache_data
def load_geojson(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_tracks(paths: list[str]) -> pd.DataFrame:

    track_path = None

    for p in paths:
        if os.path.exists(p):
            track_path = p
            break

    if track_path is None:
        st.warning("ERA5 storm-track file not found.")
        return pd.DataFrame()

    try:
        tracks = pd.read_csv(track_path, low_memory=False, encoding="utf-8")
    except Exception:
        tracks = pd.read_csv(track_path, low_memory=False, encoding="latin1")

    tracks = tracks.copy()

    rename_map = {}

    for c in tracks.columns:
        cl = c.lower()

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
        st.warning("Storm track dataset missing required columns.")
        return pd.DataFrame()

    if "value" not in tracks.columns:
        tracks["value"] = 1.0

    if "storm_id" not in tracks.columns:
        tracks["storm_id"] = tracks.groupby("year").cumcount().astype(str)

    tracks = tracks.dropna(subset=["latitude", "longitude", "year"]).copy()

    tracks["year"] = pd.to_numeric(tracks["year"], errors="coerce")
    tracks = tracks.dropna(subset=["year"]).copy()
    tracks["year"] = tracks["year"].astype(int)

    tracks["longitude"] = tracks["longitude"].apply(
        lambda x: x - 360 if x > 180 else x
    )

    return tracks


@st.cache_data
def load_capacity(paths: list[str]) -> pd.DataFrame:

    path = None
    for p in paths:
        if os.path.exists(p):
            path = p
            break

    if path is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, low_memory=False, encoding="latin1")

    df = df.copy()

    if "geopoint" in df.columns:
        lat_list = []
        lon_list = []

        for g in df["geopoint"]:
            try:
                a, b = str(g).split(",")
                lat_list.append(float(a.strip()))
                lon_list.append(float(b.strip()))
            except Exception:
                lat_list.append(np.nan)
                lon_list.append(np.nan)

        df["lat"] = lat_list
        df["lon"] = lon_list

    elif {"Postcode", "Local Authority"}.issubset(df.columns):
        df["lat"] = np.nan
        df["lon"] = np.nan

    capacity_col_candidates = [
        "Energy Source & Energy Conversion Technology 1 - Registered Capacity (MW)",
        "Already connected Registered Capacity (MW) ",
        "Maximum Export Capacity (MW)"
    ]

    for col in capacity_col_candidates:
        if col in df.columns:
            df["capacity_mw"] = pd.to_numeric(df[col], errors="coerce")
            break

    if "capacity_mw" not in df.columns:
        df["capacity_mw"] = np.nan

    if "Energy Source 1" not in df.columns:
        df["Energy Source 1"] = "Unknown"

    if "Postcode" not in df.columns:
        df["Postcode"] = "Unknown"

    if "Local Authority" not in df.columns:
        df["Local Authority"] = "Unknown"

    return df


@st.cache_data
def load_curtailment(paths: list[str]) -> pd.DataFrame:

    path = None
    for p in paths:
        if os.path.exists(p):
            path = p
            break

    if path is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, low_memory=False, encoding="latin1")

    df = df.copy()

    if "Start time UTC" in df.columns:
        df["start_ts"] = pd.to_datetime(df["Start time UTC"], errors="coerce")
        df["year"] = df["start_ts"].dt.year
    else:
        df["year"] = np.nan

    energy_col = "Outage related curtailment-Total energy reduction (MWh)"
    avg_col = "Average access reduction (MW)"

    if energy_col in df.columns:
        df["curtailment_mwh"] = pd.to_numeric(df[energy_col], errors="coerce")
    else:
        df["curtailment_mwh"] = np.nan

    if avg_col in df.columns:
        df["avg_access_mw"] = pd.to_numeric(df[avg_col], errors="coerce")
    else:
        df["avg_access_mw"] = np.nan

    if "Site" not in df.columns:
        df["Site"] = "Unknown"

    if "Reason For curtailment" not in df.columns:
        df["Reason For curtailment"] = "Unknown"

    return df


@st.cache_data
def load_feeders(paths: list[str]) -> pd.DataFrame:

    path = None
    for p in paths:
        if os.path.exists(p):
            path = p
            break

    if path is None:
        return pd.DataFrame()

    try:
        df = pd.read_csv(path, low_memory=False, encoding="utf-8")
    except Exception:
        df = pd.read_csv(path, low_memory=False, encoding="latin1")

    df = df.copy()

    if "Geo Point" in df.columns:
        lat_list = []
        lon_list = []

        for g in df["Geo Point"]:
            try:
                a, b = str(g).split(",")
                lat_list.append(float(a.strip()))
                lon_list.append(float(b.strip()))
            except Exception:
                lat_list.append(np.nan)
                lon_list.append(np.nan)

        df["lat"] = lat_list
        df["lon"] = lon_list

    df = df.dropna(subset=["lat", "lon"], how="any")

    if "Line situation" not in df.columns:
        df["Line situation"] = "Unknown"

    if "voltage" in df.columns:
        df["voltage_numeric"] = pd.to_numeric(df["voltage"], errors="coerce")
    else:
        df["voltage_numeric"] = np.nan

    return df

# =================================================
# SOURCE DATA
# =================================================

df_parent = load_parent_data(DATA_PATH)
geojson = load_geojson(GEO_PATH)
tracks = load_tracks(TRACK_CANDIDATES)
capacity = load_capacity(CAPACITY_CANDIDATES)
curtail = load_curtailment(CURTAIL_CANDIDATES)
feeders = load_feeders(FEEDER_CANDIDATES)

st.write("capacity rows:", len(capacity))
st.write("feeders rows:", len(feeders))
st.write("curtail rows:", len(curtail))

# =================================================
# REGION -> NUTS3 MAP
# =================================================

nuts_map = {
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

sub_to_parent = {
    sub: parent
    for parent, subs in nuts_map.items()
    for sub in subs
}

# =================================================
# EXPAND TO SUBREGIONS
# =================================================

@st.cache_data
def expand_to_subregions(df_parent_in: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, r in df_parent_in.iterrows():
        parent = r["region"]
        subs = nuts_map.get(parent, [])

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

    df = pd.DataFrame(rows)

    all_years = sorted(df["year"].unique())
    all_regions = sorted(df["region"].unique())

    full_index = pd.MultiIndex.from_product(
        [all_years, all_regions],
        names=["year", "region"]
    )

    df = (
        df.set_index(["year", "region"])
        .reindex(full_index)
        .reset_index()
    )

    df["parent_region"] = df["region"].map(sub_to_parent)
    df["W_mean"] = df["W_mean"].fillna(0.0)
    df["n_storm_pts"] = df["n_storm_pts"].fillna(0.0)
    df["W_mean_norm_region"] = df["W_mean_norm_region"].fillna(0.0)

    df["W_norm_year"] = df.groupby("year")["W_mean"].transform(
        lambda s: s / s.max() if s.max() > 0 else 0.0
    )

    df["W_sub_norm"] = df.groupby("region")["W_mean"].transform(
        lambda s: (s - s.min()) / (s.max() - s.min()) if s.max() > s.min() else 0.0
    )

    df["MHI"] = 0.70 * df["W_norm_year"] + 0.30 * df["W_sub_norm"]

    alpha = 1.8
    xi = 0.5 * df["W_sub_norm"] + 0.5 * df["W_norm_year"]
    df["P_fail"] = 1.0 - np.exp(-alpha * df["MHI"] * xi)

    df["Curtailment_Risk"] = np.clip(0.60 * df["MHI"] + 0.40 * df["P_fail"], 0, 1)
    df["Node_Failure_Pressure"] = np.clip(0.55 * df["P_fail"] + 0.45 * df["W_sub_norm"], 0, 1)

    return df

df = expand_to_subregions(df_parent)

# =================================================
# LOCAL GRID RISK PREP
# =================================================

@st.cache_data
def build_local_grid_risk(capacity_df: pd.DataFrame, curtail_df: pd.DataFrame, feeder_df: pd.DataFrame) -> pd.DataFrame:

    frames = []

    if not capacity_df.empty:
        cap = capacity_df.copy()
        cap["point_type"] = "Embedded Capacity"
        cap["label"] = cap["Postcode"].fillna("Unknown")
        cap["risk_score"] = pd.to_numeric(cap["capacity_mw"], errors="coerce").fillna(0)
        cap["detail"] = cap["Energy Source 1"].fillna("Unknown")
        cap = cap.dropna(subset=["lat", "lon"], how="any")
        frames.append(cap[["lat", "lon", "point_type", "label", "risk_score", "detail"]])

    if not feeder_df.empty:
        fd = feeder_df.copy()
        fd["point_type"] = "Feeder"
        fd["label"] = fd["Line situation"].fillna("Unknown")
        fd["risk_score"] = pd.to_numeric(fd["voltage_numeric"], errors="coerce").fillna(0)
        fd["detail"] = fd["Line situation"].fillna("Unknown")
        fd = fd.dropna(subset=["lat", "lon"], how="any")
        frames.append(fd[["lat", "lon", "point_type", "label", "risk_score", "detail"]])

    if not frames:
        return pd.DataFrame()

    local = pd.concat(frames, ignore_index=True)

    if local["risk_score"].max() > local["risk_score"].min():
        local["risk_norm"] = (
            local["risk_score"] - local["risk_score"].min()
        ) / (
            local["risk_score"].max() - local["risk_score"].min()
        )
    else:
        local["risk_norm"] = 0.0

    return local

local_grid = build_local_grid_risk(capacity, curtail, feeders)

# =================================================
# SIDEBAR
# =================================================

st.sidebar.header("Controls")

selected_parent = st.sidebar.selectbox(
    "Select Main Region",
    sorted(df["parent_region"].dropna().unique())
)

selected_year = st.sidebar.slider(
    "Select Year",
    int(df["year"].min()),
    int(df["year"].max()),
    int(df["year"].min())
)

scenario = st.sidebar.selectbox(
    "Scenario",
    ["Baseline", "Mild", "Gradual", "Escalation"]
)

show_both_regions = st.sidebar.checkbox("Show both main regions on map", value=True)

scenario_factor = {
    "Baseline": 1.00,
    "Mild": 1.10,
    "Gradual": 1.25,
    "Escalation": 1.50,
}[scenario]

# =================================================
# FILTERS
# =================================================

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

# =================================================
# TABS
# =================================================

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
    "Storm Animation Map",
    "ERA5 Storm Tracks",
    "Hazard Timeline",
    "Climate Risk Dashboard",
    "Scenario Simulation",
    "Storm Intensity Surface",
    "Return Period Analysis",
    "Node Failure & Curtailment",
    "Local Grid Risk",
])

# =================================================
# TAB 1 - STORM ANIMATION MAP
# =================================================

with tab1:
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
    fig_anim.update_traces(marker_line_width=0.9, marker_line_color="black")
    fig_anim.update_layout(height=650, margin=dict(l=0, r=0, t=30, b=0))

    st.plotly_chart(fig_anim, use_container_width=True)

# =================================================
# TAB 2 - ERA5 STORM TRACKS
# =================================================

with tab2:
    st.subheader("ERA5 Storm Tracks")

    if tracks.empty:
        st.warning("ERA5 storm-track file could not be found.")
    else:
        subtab1, subtab2, subtab3 = st.tabs([
            "Track Points",
            "Trajectory Lines",
            "Storm Impact Overlay",
        ])

        with subtab1:
            fig_tracks = px.scatter_mapbox(
                tracks,
                lat="latitude",
                lon="longitude",
                animation_frame="year",
                size_max=12,
                zoom=4.7,
                center={"lat": 54.5, "lon": -1.8},
                color_continuous_scale="Turbo",
                mapbox_style="carto-positron",
                hover_data={
                    "storm_id": True,
                    "year": True,
                    "latitude": ":.2f",
                    "longitude": ":.2f",
                },
            )

            fig_tracks.update_layout(height=700, margin=dict(l=0, r=0, t=10, b=0))
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
                center={"lat": 54.5, "lon": -1.8},
            )

            fig_lines.update_layout(
                mapbox_style="carto-positron",
                height=700,
                margin=dict(l=0, r=0, t=10, b=0),
                showlegend=False,
            )

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
                opacity=0.45,
                zoom=4.8,
                center={"lat": 54.5, "lon": -1.8},
                hover_data={
                    "region": True,
                    "parent_region": True,
                    "P_fail": ":.2f",
                    "MHI": ":.2f",
                },
            )

            fig_overlay.update_layout(
                mapbox_style="carto-positron",
                height=700,
                margin=dict(l=0, r=0, t=10, b=0),
            )

            fig_overlay.add_scattermapbox(
                lat=tracks_overlay["latitude"],
                lon=tracks_overlay["longitude"],
                mode="markers",
                marker=dict(size=4, opacity=0.8),
                text=tracks_overlay["storm_id"].astype(str),
                name="Storm track points",
            )

            st.plotly_chart(fig_overlay, use_container_width=True)

# =================================================
# TAB 3 - HAZARD TIMELINE
# =================================================

with tab3:
    st.subheader("Hazard Timeline")

    yearly = (
        df_selected_parent
        .groupby("year", as_index=False)
        .agg(
            W_mean=("W_mean", "mean"),
            MHI=("MHI", "mean"),
            P_fail=("P_fail", "mean"),
            Curtailment_Risk=("Curtailment_Risk", "mean"),
        )
    )

    fig_timeline = go.Figure()
    fig_timeline.add_trace(go.Scatter(x=yearly["year"], y=yearly["W_mean"], mode="lines+markers", name="Wind hazard"))
    fig_timeline.add_trace(go.Scatter(x=yearly["year"], y=yearly["MHI"], mode="lines+markers", name="MHI", yaxis="y2"))
    fig_timeline.add_trace(go.Scatter(x=yearly["year"], y=yearly["P_fail"], mode="lines+markers", name="P_fail", yaxis="y2"))
    fig_timeline.add_trace(go.Scatter(x=yearly["year"], y=yearly["Curtailment_Risk"], mode="lines+markers", name="Curtailment risk", yaxis="y2"))

    fig_timeline.update_layout(
        height=560,
        xaxis_title="Year",
        yaxis=dict(title="Absolute hazard"),
        yaxis2=dict(title="Relative index / risk", overlaying="y", side="right", range=[0, 1.05]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        margin=dict(l=20, r=20, t=20, b=20),
    )

    st.plotly_chart(fig_timeline, use_container_width=True)

# =================================================
# TAB 4 - CLIMATE RISK DASHBOARD
# =================================================

with tab4:
    st.subheader("Climate Risk Dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Selected year", selected_year)
    c2.metric("Mean hazard", f"{df_selected_year['W_mean'].mean():.2f}")
    c3.metric("Mean MHI", f"{df_selected_year['MHI'].mean():.2f}")
    c4.metric("Mean P_fail", f"{df_selected_year['P_fail'].mean():.2f}")

    left, right = st.columns([1.25, 1])

    with left:
        st.markdown("#### Grid failure risk map")

        fig_risk = px.choropleth(
            df_map_year,
            geojson=geojson,
            locations="region",
            featureidkey="properties.NUTS_NAME",
            color="P_fail_scenario",
            color_continuous_scale="RdYlBu_r",
            range_color=[0, 1],
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
        fig_risk.update_traces(marker_line_width=0.9, marker_line_color="black")
        fig_risk.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))

        st.plotly_chart(fig_risk, use_container_width=True)

    with right:
        st.markdown("#### MHI digital twin panel")

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
            color_continuous_scale="Turbo",
            range_color=[0, 1],
        )

        fig_mhi_bar.update_layout(height=560, yaxis_title="Subregion", xaxis_title="MHI", margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_mhi_bar, use_container_width=True)

    st.markdown("#### Relative hazard heatmap")

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
    )

    fig_heat.update_layout(height=500, coloraxis_colorbar_title="Relative severity", xaxis_title="Year", yaxis_title="Subregion")
    st.plotly_chart(fig_heat, use_container_width=True)

# =================================================
# TAB 5 - SCENARIO SIMULATION
# =================================================

with tab5:
    st.subheader("Scenario Simulation")

    scenario_df = (
        df_selected_parent
        .groupby("year", as_index=False)
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
        temp["Curtailment_scenario"] = np.clip(0.60 * temp["MHI_scenario"] + 0.40 * temp["P_fail_scenario"], 0, 1)
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
        fig_scen_mhi.update_layout(height=480, xaxis_title="Year", yaxis_title="Scenario MHI")
        st.plotly_chart(fig_scen_mhi, use_container_width=True)

    with col_b:
        fig_scen_pf = px.line(
            scen_all,
            x="year",
            y="P_fail_scenario",
            color="Scenario",
            markers=True,
        )
        fig_scen_pf.update_layout(height=480, xaxis_title="Year", yaxis_title="Scenario P_fail")
        st.plotly_chart(fig_scen_pf, use_container_width=True)

    fig_scen_curt = px.line(
        scen_all,
        x="year",
        y="Curtailment_scenario",
        color="Scenario",
        markers=True,
    )
    fig_scen_curt.update_layout(height=420, xaxis_title="Year", yaxis_title="Scenario curtailment risk")
    st.plotly_chart(fig_scen_curt, use_container_width=True)

# =================================================
# TAB 6 - STORM INTENSITY SURFACE
# =================================================

with tab6:
    st.subheader("Storm Intensity Surface Map")

    if tracks.empty:
        st.warning("ERA5 storm-track file could not be found.")
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
            radius=18,
            center={"lat": 54.5, "lon": -1.8},
            zoom=4.8,
            mapbox_style="carto-positron",
            color_continuous_scale="Turbo",
        )

        fig_surface.update_layout(height=700, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_surface, use_container_width=True)

# =================================================
# TAB 7 - RETURN PERIOD ANALYSIS
# =================================================

with tab7:
    st.subheader("Return Period Analysis")

    st.caption("This panel uses an empirical extreme-value style return-period estimate from annual maxima.")

    rp_df = (
        df_selected_parent
        .groupby("year", as_index=False)
        .agg(Annual_Max_Hazard=("W_mean", "max"))
        .sort_values("Annual_Max_Hazard", ascending=False)
        .reset_index(drop=True)
    )

    n = len(rp_df)
    rp_df["rank"] = np.arange(1, n + 1)
    rp_df["exceedance_prob"] = rp_df["rank"] / (n + 1)
    rp_df["return_period"] = 1.0 / rp_df["exceedance_prob"]

    col1, col2 = st.columns(2)

    with col1:
        fig_rp = px.scatter(
            rp_df,
            x="return_period",
            y="Annual_Max_Hazard",
            hover_data={"year": True},
            log_x=True,
        )
        fig_rp.update_layout(height=480, xaxis_title="Return period (years, log scale)", yaxis_title="Annual maximum hazard")
        st.plotly_chart(fig_rp, use_container_width=True)

    with col2:
        fig_rank = px.bar(
            rp_df.sort_values("year"),
            x="year",
            y="Annual_Max_Hazard",
        )
        fig_rank.update_layout(height=480, xaxis_title="Year", yaxis_title="Annual maximum hazard")
        st.plotly_chart(fig_rank, use_container_width=True)

# =================================================
# TAB 8 - NODE FAILURE & CURTAILMENT
# =================================================

with tab8:
    st.subheader("Power Grid Node Failure Simulation and Energy Curtailment Risk")

    current_parent = df_selected_year.copy()

    left, right = st.columns(2)

    with left:
        st.markdown("#### Node failure simulation")

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
        fig_node.update_traces(marker_line_width=0.9, marker_line_color="black")
        fig_node.update_layout(height=560, margin=dict(l=0, r=0, t=10, b=0))
        st.plotly_chart(fig_node, use_container_width=True)

    with right:
        st.markdown("#### Energy curtailment risk")

        fig_curt = px.bar(
            current_parent.sort_values("Curtailment_Risk", ascending=False),
            x="Curtailment_Risk",
            y="region",
            orientation="h",
            color="Curtailment_Risk",
            color_continuous_scale="OrRd",
            range_color=[0, 1],
        )

        fig_curt.update_layout(height=560, xaxis_title="Curtailment risk", yaxis_title="Subregion", coloraxis_showscale=False)
        st.plotly_chart(fig_curt, use_container_width=True)

    summary = current_parent[["region", "Node_Failure_Pressure", "Curtailment_Risk", "P_fail", "MHI"]].sort_values(
        "Curtailment_Risk", ascending=False
    )
    st.markdown("#### Subregional summary")
    st.dataframe(summary, use_container_width=True)

# =================================================
# TAB 9 - LOCAL GRID RISK
# =================================================

with tab9:

    st.subheader("Local Grid Risk (postcode / feeders / curtailment)")

    st.write("capacity rows:", len(capacity))
    st.write("feeders rows:", len(feeders))
    st.write("curtail rows:", len(curtail))

    c1, c2, c3 = st.columns(3)
    c1.metric("Embedded generation sites", len(capacity))
    c2.metric("Curtailment events", len(curtail))
    c3.metric("Grid feeders", len(feeders))

    subtab1, subtab2, subtab3, subtab4 = st.tabs([
        "Embedded Capacity",
        "Feeder Network",
        "Curtailment Timeline",
        "Local Risk Overlay",
    ])

    with subtab1:
        st.markdown("#### Embedded generation (postcode locations)")

        if capacity.empty or capacity["lat"].dropna().empty:
            st.info("Embedded capacity dataset found, but no usable geopoint rows were parsed.")
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
                center={"lat": 54.5, "lon": -1.8},
                mapbox_style="carto-positron",
            )
            fig_cap.update_layout(height=650, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_cap, use_container_width=True)

    with subtab2:
        st.markdown("#### Grid feeder network")

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
                    "voltage": True,
                },
                zoom=5,
                center={"lat": 54.5, "lon": -1.8},
                mapbox_style="carto-positron",
            )
            fig_feed.update_layout(height=650, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_feed, use_container_width=True)

    with subtab3:
        st.markdown("#### Curtailment timeline and hotspots")

        if curtail.empty:
            st.info("Curtailment dataset not available.")
        else:
            cur_year = (
                curtail.groupby("year", as_index=False)
                .agg(
                    events=("Event ID", "count"),
                    curtailment_mwh=("curtailment_mwh", "sum"),
                )
            )

            col_a, col_b = st.columns(2)

            with col_a:
                fig_cur_events = px.line(cur_year, x="year", y="events", markers=True)
                fig_cur_events.update_layout(height=400, xaxis_title="Year", yaxis_title="Curtailment events")
                st.plotly_chart(fig_cur_events, use_container_width=True)

            with col_b:
                fig_cur_mwh = px.line(cur_year, x="year", y="curtailment_mwh", markers=True)
                fig_cur_mwh.update_layout(height=400, xaxis_title="Year", yaxis_title="Curtailment energy (MWh)")
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
            fig_site.update_layout(height=600, xaxis_title="Total curtailed energy (MWh)", yaxis_title="Site")
            st.plotly_chart(fig_site, use_container_width=True)

    with subtab4:
        st.markdown("#### Future local outage / curtailment risk")

        if local_grid.empty:
            st.info("Local capacity / feeder points not available.")
        else:
            local = local_grid.copy()

            local["future_risk"] = np.clip(
                0.45 * local["risk_norm"] + 0.35 * scenario_factor / 1.5 + 0.20 * (selected_year - df["year"].min()) / max(1, (df["year"].max() - df["year"].min())),
                0,
                1
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
                center={"lat": 54.5, "lon": -1.8},
                mapbox_style="carto-positron",
            )

            fig_local.update_layout(height=700, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_local, use_container_width=True)
