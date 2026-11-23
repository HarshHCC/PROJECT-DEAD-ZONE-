import math
import os

import folium
from folium.plugins import HeatMap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from streamlit_folium import st_folium

CSV_PATH = "signals.csv"
ESRI_TILES = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Street_Map/MapServer/tile/{z}/{y}/{x}"
)

PRESETS = {
    "Custom / Map Click": None,
    "Central / CP Area (28.6200, 77.2150)": (28.6200, 77.2150),
    "Connaught Place (28.6315, 77.2167)": (28.6315, 77.2167),
    "India Gate (28.6129, 77.2295)": (28.6129, 77.2295),
    "Karol Bagh (28.6519, 77.1909)": (28.6519, 77.1909),
}

st.set_page_config(page_title="Project Dead Zone", layout="wide")
st.title("Project Dead Zone — AI Signal & Crowdsourced Map")
st.caption(
    "Click the map or enter coordinates to predict dead zones, find the nearest"
    " good signal, filter readings, and log crowdsourced data."
)


def haversine_km(lat1, lon1, lat2, lon2):
  r = 6371.0
  p1, p2 = math.radians(lat1), math.radians(lat2)
  dp = math.radians(lat2 - lat1)
  dl = math.radians(lon2 - lon1)
  a = (
      math.sin(dp / 2) ** 2
      + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
  )
  return 2 * r * math.asin(math.sqrt(a))


@st.cache_data
def load_data(path, _version):
  df = pd.read_csv(path, on_bad_lines="skip")
  for col in ["lat", "lng", "signal_dbm", "speed_mbps", "is_dead_zone"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")
  df = df.dropna(subset=["lat", "lng", "is_dead_zone"]).reset_index(drop=True)
  df["is_dead_zone"] = df["is_dead_zone"].astype(int)
  return df


def train_model(df):
  model = RandomForestClassifier(n_estimators=150, random_state=42)
  model.fit(df[["lat", "lng"]], df["is_dead_zone"])
  return model


if "data_version" not in st.session_state:
  st.session_state.data_version = 0
if "check_lat" not in st.session_state:
  st.session_state.check_lat = 28.6200
if "check_lng" not in st.session_state:
  st.session_state.check_lng = 77.2150

if not os.path.exists(CSV_PATH):
  st.error(f"Missing `{CSV_PATH}` in the project folder.")
  st.stop()

df = load_data(CSV_PATH, st.session_state.data_version)
if len(df) < 2:
  st.warning("Need at least 2 valid rows in `signals.csv` to train the model.")
  st.stop()

model = train_model(df)

# Sidebar controls
st.sidebar.header("1. Location & Presets")
preset_choice = st.sidebar.selectbox("Quick Area Preset", list(PRESETS.keys()))
if preset_choice != "Custom / Map Click" and PRESETS[preset_choice]:
  st.session_state.check_lat, st.session_state.check_lng = PRESETS[
      preset_choice
  ]

check_lat = st.sidebar.number_input(
    "Your Latitude",
    value=float(st.session_state.check_lat),
    format="%.4f",
    key="lat_input",
)
check_lng = st.sidebar.number_input(
    "Your Longitude",
    value=float(st.session_state.check_lng),
    format="%.4f",
    key="lng_input",
)
st.session_state.check_lat = check_lat
st.session_state.check_lng = check_lng

st.sidebar.header("2. Map Filters & Layers")
status_filter = st.sidebar.radio(
    "Show Readings",
    ["All", "Dead Zones Only", "Good Signals Only"],
    index=0,
)
min_speed = st.sidebar.slider(
    "Minimum Speed Filter (Mbps)",
    min_value=0.0,
    max_value=float(max(10.0, df["speed_mbps"].max())),
    value=0.0,
    step=1.0,
)
show_markers = st.sidebar.checkbox("Show Signal Markers", value=True)
show_heatmap = st.sidebar.checkbox("Show Signal Heatmap Overlay", value=True)
show_route = st.sidebar.checkbox("Show Route to Nearest Good Spot", value=True)

# Apply table/marker filters
filtered_df = df[df["speed_mbps"] >= min_speed].copy()
if status_filter == "Dead Zones Only":
  filtered_df = filtered_df[filtered_df["is_dead_zone"] == 1]
elif status_filter == "Good Signals Only":
  filtered_df = filtered_df[filtered_df["is_dead_zone"] == 0]

# AI prediction & probabilities
probs = model.predict_proba([[check_lat, check_lng]])[0]
prob_good = float(probs[0]) if len(probs) > 1 else 0.0
prob_dead = float(probs[1]) if len(probs) > 1 else float(probs[0])
pred_dead = 1 if prob_dead >= 0.5 else 0
confidence = prob_dead if pred_dead == 1 else prob_good

# Nearest good signal (from full dataset so routing stays helpful)
good_spots = df[df["is_dead_zone"] == 0].copy()
nearest_spot = None
nearest_dist = None
if not good_spots.empty:
  good_spots["dist_km"] = good_spots.apply(
      lambda r: haversine_km(check_lat, check_lng, r["lat"], r["lng"]), axis=1
  )
  nearest_spot = good_spots.loc[good_spots["dist_km"].idxmin()]
  nearest_dist = float(nearest_spot["dist_km"])

# Top KPI row
dead_count = int((df["is_dead_zone"] == 1).sum())
good_count = int((df["is_dead_zone"] == 0).sum())

c1, c2, c3, c4 = st.columns(4)
c1.metric(
    "AI Prediction Here",
    "DEAD ZONE" if pred_dead == 1 else "GOOD SIGNAL",
    f"{confidence * 100:.0f}% confidence",
)
if nearest_spot is not None:
  c2.metric(
      "Nearest Good Signal",
      f"{nearest_dist:.2f} km away",
      f"{nearest_spot['speed_mbps']:.1f} Mbps",
  )
else:
  c2.metric("Nearest Good Signal", "None logged yet")
c3.metric("Total Readings", f"{len(df)} points", f"{good_count} good / {dead_count} dead")
c4.metric(
    "Avg Speed",
    f"{df['speed_mbps'].mean():.1f} Mbps",
    f"Avg {df['signal_dbm'].mean():.0f} dBm",
)

# Probability breakdown bar
st.progress(
    prob_dead,
    text=(
        f"Dead Zone Probability: {prob_dead * 100:.1f}%  |  Good Signal"
        f" Probability: {prob_good * 100:.1f}%"
    ),
)

# Map + Nearest Spot Details side-by-side
map_col, info_col = st.columns([2.2, 1])

with map_col:
  m = folium.Map(location=[check_lat, check_lng], zoom_start=13, tiles=None)
  folium.TileLayer(
      tiles=ESRI_TILES, attr="Esri", name="World Street Map"
  ).add_to(m)

  if show_heatmap and not df.empty:
    heat_data = [
        [
            r["lat"],
            r["lng"],
            0.25
            if r["is_dead_zone"] == 1
            else max(0.45, min(1.0, r["speed_mbps"] / 80.0)),
        ]
        for _, r in df.iterrows()
    ]
    HeatMap(heat_data, radius=22, blur=16).add_to(m)

  if show_markers:
    for _, r in filtered_df.iterrows():
      is_dz = r["is_dead_zone"] == 1
      color = "red" if is_dz else "green"
      folium.CircleMarker(
          location=[r["lat"], r["lng"]],
          radius=7,
          color=color,
          weight=2,
          fill=True,
          fill_opacity=0.8,
          tooltip=(
              f"{'Dead Zone' if is_dz else 'Good Signal'}: "
              f"{r['speed_mbps']:.1f} Mbps ({r['signal_dbm']:.0f} dBm)"
          ),
          popup=folium.Popup(
              f"<b>{'DEAD ZONE' if is_dz else 'GOOD SIGNAL'}</b><br>"
              f"Coords: ({r['lat']:.4f}, {r['lng']:.4f})<br>"
              f"Signal: {r['signal_dbm']:.1f} dBm<br>"
              f"Speed: {r['speed_mbps']:.1f} Mbps",
              max_width=240,
          ),
      ).add_to(m)

  folium.Marker(
      location=[check_lat, check_lng],
      tooltip="Selected Location (Click map to move)",
      popup=f"Selected: ({check_lat:.4f}, {check_lng:.4f})",
      icon=folium.Icon(
          color="orange" if pred_dead == 1 else "blue",
          icon="info-sign",
      ),
  ).add_to(m)

  if nearest_spot is not None and show_route:
    folium.Marker(
        location=[nearest_spot["lat"], nearest_spot["lng"]],
        tooltip=f"Nearest Good Spot ({nearest_dist:.2f} km)",
        popup=(
            f"<b>Nearest Good Signal</b><br>"
            f"Distance: {nearest_dist:.2f} km<br>"
            f"Speed: {nearest_spot['speed_mbps']:.1f} Mbps<br>"
            f"Signal: {nearest_spot['signal_dbm']:.1f} dBm"
        ),
        icon=folium.Icon(color="green", icon="ok-sign"),
    ).add_to(m)
    folium.PolyLine(
        locations=[
            [check_lat, check_lng],
            [nearest_spot["lat"], nearest_spot["lng"]],
        ],
        color="#1f77b4",
        weight=4,
        dash_array="8, 6",
        tooltip=f"Straight-line distance: {nearest_dist:.2f} km",
    ).add_to(m)

  map_event = st_folium(
      m,
      width=None,
      height=480,
      returned_objects=["last_clicked"],
  )

  # Update coordinates if user clicks somewhere new on the map
  if map_event and map_event.get("last_clicked"):
    clicked_lat = round(float(map_event["last_clicked"]["lat"]), 4)
    clicked_lng = round(float(map_event["last_clicked"]["lng"]), 4)
    if (
        abs(clicked_lat - st.session_state.check_lat) > 1e-4
        or abs(clicked_lng - st.session_state.check_lng) > 1e-4
    ):
      st.session_state.check_lat = clicked_lat
      st.session_state.check_lng = clicked_lng
      st.rerun()

with info_col:
  st.subheader("Spot & Route Details")
  st.write(f"**Checked Coords:** `{check_lat:.4f}, {check_lng:.4f}`")
  st.caption("Tip: Click any point on the map to test that spot.")

  if nearest_spot is not None:
    walk_mins = max(1, round((nearest_dist / 4.8) * 60))
    drive_mins = max(1, round((nearest_dist / 25.0) * 60))
    st.markdown("---")
    st.markdown("**Recommended Good Spot**")
    st.write(
        f"- **Coordinates:** `{nearest_spot['lat']:.4f}, {nearest_spot['lng']:.4f}`"
    )
    st.write(f"- **Straight-Line Distance:** `{nearest_dist:.2f} km`")
    st.write(f"- **Est. Time:** ~`{walk_mins} min` walk / ~`{drive_mins} min` drive")
    st.write(
        f"- **Measured Performance:** `{nearest_spot['speed_mbps']:.1f} Mbps` "
        f"(`{nearest_spot['signal_dbm']:.0f} dBm`)"
    )
  else:
    st.info("No good signal points (`is_dead_zone == 0`) found yet.")

# Crowdsourced submission form pre-filled with current coordinates
st.markdown("---")
st.subheader("Submit a New Crowdsourced Signal Reading")
with st.form("submit_signal_form", clear_on_submit=False):
  f1, f2, f3, f4 = st.columns(4)
  new_lat = f1.number_input("Latitude", value=float(check_lat), format="%.4f")
  new_lng = f2.number_input("Longitude", value=float(check_lng), format="%.4f")
  new_dbm = f3.number_input("Signal (dBm)", value=-112.0, format="%.1f")
  new_speed = f4.number_input("Speed (Mbps)", value=0.8, format="%.1f")
  submitted = st.form_submit_button("Save & Retrain AI")

if submitted:
  label = 1 if (new_dbm < -105 or new_speed < 2.0) else 0
  if os.path.getsize(CSV_PATH) > 0:
    with open(CSV_PATH, "rb") as f:
      f.seek(-1, os.SEEK_END)
      if f.read(1) not in (b"\n", b"\r"):
        with open(CSV_PATH, "ab") as fa:
          fa.write(b"\n")
  row_df = pd.DataFrame([{
      "lat": new_lat,
      "lng": new_lng,
      "signal_dbm": new_dbm,
      "speed_mbps": new_speed,
      "is_dead_zone": label,
  }])
  row_df.to_csv(CSV_PATH, mode="a", header=False, index=False)
  st.session_state.data_version += 1
  st.success(
      f"Saved reading at ({new_lat:.4f}, {new_lng:.4f}) as "
      f"{'DEAD ZONE' if label == 1 else 'GOOD SIGNAL'} and retrained the model!"
  )
  st.rerun()

# Filtered data table & CSV export
with st.expander(f"Crowdsourced Dataset ({len(filtered_df)} matching rows)"):
  st.dataframe(filtered_df, use_container_width=True)
  st.download_button(
      label="Download Full signals.csv",
      data=df.to_csv(index=False).encode("utf-8"),
      file_name="signals.csv",
      mime="text/csv",
  )