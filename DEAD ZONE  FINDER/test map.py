import folium

m = folium.Map(location=[28.6139, 77.2090], zoom_start=12, tiles=None)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="World Street Map",
).add_to(m)

folium.Marker(
    location=[28.6139, 77.2090],
    popup="Connaught Place, Delhi",
    tooltip="Click me",
).add_to(m)

m.save("first_marker.html")
print("Updated first_marker.html")