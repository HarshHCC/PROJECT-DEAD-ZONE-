import folium
import pandas as pd

df = pd.read_csv("signals.csv")

m = folium.Map(location=[28.6139, 77.2090], zoom_start=13, tiles=None)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="World Street Map",
).add_to(m)

for _, row in df.iterrows():
    color = "red" if row["is_dead_zone"] == 1 else "green"
    status = "Dead Zone" if row["is_dead_zone"] == 1 else "Good Signal"

    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=9,
        color=color,
        fill=True,
        fill_opacity=0.8,
        popup=f"{status} | {row['signal_dbm']} dBm | {row['speed_mbps']} Mbps",
        tooltip=status,
    ).add_to(m)

m.save("signals_map.html")
print("Saved signals_map.html")