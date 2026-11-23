import os
import folium
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

csv_path = "signals.csv"

print("--- Submit New Signal Data ---")
lat = float(input("Enter Latitude (e.g., 28.6220): "))
lng = float(input("Enter Longitude (e.g., 77.2120): "))
signal_dbm = float(input("Enter Signal Strength in dBm (e.g., -110): "))
speed_mbps = float(input("Enter Speed in Mbps (e.g., 1.5): "))

is_dead_zone = 1 if (signal_dbm < -105 or speed_mbps < 2.0) else 0

if os.path.exists(csv_path) and os.path.getsize(csv_path) > 0:
  with open(csv_path, "rb") as f:
    f.seek(-1, os.SEEK_END)
    last_char = f.read(1)
  if last_char not in (b"\n", b"\r"):
    with open(csv_path, "ab") as f:
      f.write(b"\n")

new_entry = pd.DataFrame([{
    "lat": lat,
    "lng": lng,
    "signal_dbm": signal_dbm,
    "speed_mbps": speed_mbps,
    "is_dead_zone": is_dead_zone,
}])

new_entry.to_csv(csv_path, mode="a", header=False, index=False)
status = "DEAD ZONE" if is_dead_zone == 1 else "GOOD SIGNAL"
print(f"Added to signals.csv! Classed as: {status}")
df = pd.read_csv(csv_path, on_bad_lines="skip")
X = df[["lat", "lng"]]
y = df["is_dead_zone"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

m = folium.Map(location=[28.6139, 77.2090], zoom_start=13, tiles=None)
folium.TileLayer(
    tiles=(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}"
    ),
    attr="Esri",
    name="World Street Map",
).add_to(m)

for _, row in df.iterrows():
  color = "red" if row["is_dead_zone"] == 1 else "green"
  folium.CircleMarker(
      location=[row["lat"], row["lng"]],
      radius=7,
      color=color,
      fill=True,
      fill_opacity=0.7,
      popup=f"Signal: {row['signal_dbm']} dBm<br>Speed: {row['speed_mbps']} Mbps",
  ).add_to(m)

m.save("updated_crowdsourced_map.html")
print("Saved updated_crowdsourced_map.html!")