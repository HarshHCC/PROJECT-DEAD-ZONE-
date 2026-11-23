import math
import folium
import pandas as pd


def distance_km(lat1, lng1, lat2, lng2):
    """Calculate straight-line distance in km between two GPS points (Haversine formula)."""
    r = 6371.0  # Earth's radius in kilometers
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lng2 - lng1)

    a = (
        math.sin(dp / 2) ** 2
        + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    )
    return 2 * r * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# 1. Load signal readings and keep only Good Signal rows (is_dead_zone == 0)
df = pd.read_csv("signals.csv")
good_spots = df[df["is_dead_zone"] == 0].copy()

# 2. Suppose a user is stuck in a dead zone at this coordinate
user_lat, user_lng = 28.6310, 77.2210

# 3. Find the closest good-signal spot
best_spot = None
best_dist = float("inf")

for _, row in good_spots.iterrows():
    d = distance_km(user_lat, user_lng, row["lat"], row["lng"])
    if d < best_dist:
        best_dist = d
        best_spot = row

print(f"Closest Good Signal: ({best_spot['lat']}, {best_spot['lng']})")
print(f"Distance: {best_dist:.2f} km | Speed: {best_spot['speed_mbps']} Mbps")

# 4. Plot user location, nearest good spot, and a connecting line on a map
m = folium.Map(location=[user_lat, user_lng], zoom_start=14, tiles=None)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="World Street Map",
).add_to(m)

# User's current spot (red marker)
folium.Marker(
    location=[user_lat, user_lng],
    popup="You Are Here (Dead Zone)",
    tooltip="Current Spot",
    icon=folium.Icon(color="red", icon="remove-sign"),
).add_to(m)

# Nearest good signal spot (green marker)
folium.Marker(
    location=[best_spot["lat"], best_spot["lng"]],
    popup=f"Nearest Good Signal ({best_dist:.2f} km away)<br>Speed: {best_spot['speed_mbps']} Mbps",
    tooltip="Nearest Good Signal",
    icon=folium.Icon(color="green", icon="ok-sign"),
).add_to(m)

# Line connecting user to nearest good signal
folium.PolyLine(
    locations=[[user_lat, user_lng], [best_spot["lat"], best_spot["lng"]]],
    color="blue",
    weight=4,
    dash_array="6",
).add_to(m)

m.save("nearest_signal_map.html")
print("Saved nearest_signal_map.html")