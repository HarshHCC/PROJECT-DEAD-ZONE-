# PROJECT-DEAD-ZONE-
# 📡 Project Dead Zone — AI Signal Predictor & Crowdsourced Network Map

An interactive, crowdsourced geospatial application built with Python. **Project Dead Zone** maps network signal strength and data speeds across location points, predicts cellular dead zones using a **Random Forest** machine learning model, and routes users to the nearest high-performance signal location using the **Haversine formula**.

---

## 🚀 Key Features

* **AI Dead-Zone Prediction:** Trains a Random Forest Classifier on location parameters to output real-time dead zone probabilities and predictions.
* **Interactive Map & Heatmap:** Built with Folium and Esri World Street Map tiles to visualize network coverage markers and signal heat density overlays.
* **Nearest Good Signal Router:** Calculates straight-line distance and estimates travel times (walking/driving) to the closest verified high-speed signal point.
* **Crowdsourced Data Collection:** Live submission interface to log network readings (`lat`, `lng`, `signal_dbm`, `speed_mbps`) into `signals.csv` with instant model retraining.
* **Integrated Analytics:** Built-in dashboard visualizations comparing signal strength against download speed distribution.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Web Framework:** Streamlit (`streamlit-folium`)
* **Machine Learning & Data Processing:** Scikit-Learn, Pandas, NumPy
* **Geospatial & Mapping:** Folium, Esri World Street Map Tiles

---

## 📦 Project Structure

```text
DEAD ZONE FINDER/
│-- app.py                  # Main interactive Streamlit application
│-- signals.csv             # Crowdsourced signal dataset
│-- requirements.txt        # Python dependency specifications
│-- README.md               # Project documentation
│-- .gitignore              # Ignored files for version control
│-- modular_scripts/        # Legacy standalone Python scripts
    │-- add_signal.py
    │-- ai preditor.py
    │-- nearest_good_signal.py
    │-- plotsignals.py
    │-- test map.py
