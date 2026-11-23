import folium
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# 1. Load data
df = pd.read_csv("signals.csv")

# 2. Features (inputs) and Target (output)
X = df[["lat", "lng"]]  # Location coordinates
y = df["is_dead_zone"]  # 0 = Good Signal, 1 = Dead Zone

# 3. Train the AI Model (Random Forest Classifier)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)
print("AI Model Trained Successfully!")

# 4. Give the AI 3 brand-new unmeasured coordinates in Delhi to test
test_locations = [
    {"name": "Spot A (Near Connaught Place)", "lat": 28.6145, "lng": 77.2095},
    {"name": "Spot B (Near Old Rohtak Road)", "lat": 28.6310, "lng": 77.2210},
    {"name": "Spot C (Near Pusa Road)", "lat": 28.6060, "lng": 77.2010},
]

# 5. Create Map
m = folium.Map(location=[28.6139, 77.2090], zoom_start=13, tiles=None)

folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Street_Map/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="World Street Map",
).add_to(m)

# Plot actual historical points (Circles)
for _, row in df.iterrows():
    color = "red" if row["is_dead_zone"] == 1 else "green"
    folium.CircleMarker(
        location=[row["lat"], row["lng"]],
        radius=7,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=f"Known Point: {'Dead Zone' if row['is_dead_zone'] == 1 else 'Good Signal'}",
    ).add_to(m)

# Predict & Plot AI test spots (Star Markers)
for spot in test_locations:
    # Ask AI to predict: 0 or 1
    prediction = model.predict([[spot["lat"], spot["lng"]]])[0]
    result_text = "AI Predicts: DEAD ZONE" if prediction == 1 else "AI Predicts: GOOD SIGNAL"
    marker_color = "red" if prediction == 1 else "green"

    folium.Marker(
        location=[spot["lat"], spot["lng"]],
        popup=f"{spot['name']}<br><b>{result_text}</b>",
        tooltip=f"{spot['name']} ({result_text})",
        icon=folium.Icon(color=marker_color, icon="info-sign"),
    ).add_to(m)

# 6. Save final map
m.save("ai_predictions_map.html")
print("Saved ai_predictions_map.html")