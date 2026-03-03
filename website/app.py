import streamlit as st
import pandas as pd
import plotly.express as px
import json
import os

st.set_page_config(layout="wide", page_title="Regional Wind Hazard Dashboard")

st.title("Regional Wind Hazard Dashboard")
st.markdown("Storm-track buffer based regional wind severity")

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, "data", "hazard_final.parquet")
GEO_PATH = os.path.join(BASE_DIR, "geo", "regions_NE_Yorkshire.geojson")

@st.cache_data
def load_data():
    return pd.read_parquet(DATA_PATH)

df = load_data()

st.sidebar.header("Controls")

region = st.sidebar.selectbox("Select Region", sorted(df["region"].unique()))

year = st.sidebar.slider(
    "Select Year",
    int(df["year"].min()),
    int(df["year"].max()),
    int(df["year"].min())
)

df_filtered = df[(df["region"] == region) & (df["year"] == year)].copy()

st.subheader("Wind Hazard Map (Year-scaled comparison)")

with open(GEO_PATH) as f:
    geojson = json.load(f)

df_year = df[df["year"] == year].copy()

if df_year["W_mean"].max() > 0:
    df_year["W_norm_year"] = df_year["W_mean"] / df_year["W_mean"].max()
else:
    df_year["W_norm_year"] = 0

fig_map = px.choropleth(
    df_year,
    geojson=geojson,
    locations="region",
    featureidkey="properties.nuts118nm",
    color="W_norm_year",
    range_color=[0, 1],
    color_continuous_scale="YlOrRd",
    hover_data={
        "region": True,
        "year": True,
        "W_mean": ":.2f",
        "n_storm_pts": True,
    }
)

fig_map.update_geos(fitbounds="locations", visible=False)
st.plotly_chart(fig_map, use_container_width=True)

st.subheader("Trend Over Time (Absolute Severity)")

df_region = df[df["region"] == region]

fig_trend = px.line(df_region, x="year", y="W_mean", markers=True)
st.plotly_chart(fig_trend, use_container_width=True)

st.subheader("Heatmap (Relative Within Each Region)")

heat = df.pivot(index="region", columns="year", values="W_mean_norm_region")

fig_heat = px.imshow(
    heat,
    aspect="auto",
    color_continuous_scale="YlOrRd",
    zmin=0,
    zmax=1
)

st.plotly_chart(fig_heat, use_container_width=True)

# ---------------- HEATMAP ----------------
st.subheader("Heatmap (Relative Within Each Region)")

heat = df.pivot(
    index="region",
    columns="year",
    values="W_mean_norm_region"
)

fig_heat = px.imshow(
    heat,
    aspect="auto",
    color_continuous_scale="YlOrRd",
    zmin=0,
    zmax=1
)

fig_heat.update_layout(
    coloraxis_colorbar_title="Relative Severity (0–1)",
    xaxis_title="Year",
    yaxis_title="Region"
)

st.plotly_chart(fig_heat, use_container_width=True)