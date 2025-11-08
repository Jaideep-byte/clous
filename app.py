from flask import Flask, render_template, jsonify, request
import os, pickle, threading, subprocess, json, time, requests
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient, DESCENDING
from notifier import send_alert
import numpy as np
from bson.objectid import ObjectId

# --- System Constants ---
ML_DIR = "ml"
MODEL_DIR = os.path.join(ML_DIR, "ml_models")

# This is the feature order your model was trained on.
FEATURES = ["T", "H", "Soil", "Rain", "Flame", "Vib", "P", "Alt"]

# --- Locks ---
model_lock = threading.Lock()

# --- Secrets from Render Environment Variables ---
MONGODB_URI = os.environ.get("MONGODB_ATLAS_URI")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")
# NEW: Get Google API Key from environment
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY") 

if not MONGODB_URI or not API_SECRET_KEY:
    print("FATAL ERROR: MONGODB_ATLAS_URI or API_SECRET_KEY not set in environment.")

if not GOOGLE_API_KEY:
    print("WARNING: GOOGLE_API_KEY not set. Weather forecast will not work.")
    
# --- MongoDB Atlas Setup ---
try:
    if MONGODB_URI is None:
        raise ValueError("MONGODB_ATLAS_URI environment variable not found.")
        
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client["DisasterDB"]
    collection = db["SensorData"]
    print("Connected to MongoDB Atlas.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    exit()

# --- Flask setup ---
app = Flask(__name__)

# --- Global State ---
model, time_model, scaler, le = None, None, None, None
previous_risk_state = "None"

# --- Model Loading ---
def load_models():
    global model, time_model, scaler, le, model_lock
    try:
        with model_lock:
            print(f"Attempting to load models from {MODEL_DIR}...")
            model = pickle.load(open(os.path.join(MODEL_DIR, "disaster_model.pkl"), "rb"))
            time_model = pickle.load(open(os.path.join(MODEL_DIR, "time_predictor.pkl"), "rb"))
            scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
            le = pickle.load(open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb"))
            print("Models loaded successfully.")
    except Exception as e:
        print(f"FATAL: Model files not found in {MODEL_DIR}. {e}")

# --- (All other helper functions from your script) ---
# --- REMOVED fetch_weather_alerts() function ---

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

@app.route('/data')
def get_data():
    try:
        data_cursor = collection.find({}, {"_id": 0}).sort("timestamp", DESCENDING).limit(20)
        data = list(data_cursor); data.reverse()
        for item in data: item['timestamp'] = item['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        return jsonify(data)
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/historical_data')
def get_historical_data():
    try:
        hours = 24 * 7 if request.args.get('period') == '7d' else 24
        start_time = datetime.now() - timedelta(hours=hours)
        pipeline = [
            {'$match': { 'timestamp': { '$gte': start_time } }},
            {'$group': {
                '_id': {'$dateTrunc': { 'date': "$timestamp", 'unit': 'hour' }},
                'avg_T': { '$avg': '$T' }, 'avg_H': { '$avg': '$H' },
                'avg_Soil': { '$avg': '$Soil' }, 'total_Rain': { '$sum': '$Rain' }
            }},
            {'$sort': { '_id': 1 }}
        ]
        results = list(collection.aggregate(pipeline))
        formatted_results = {
            'labels': [r['_id'].strftime('%Y-%m-%dT%H:%M:%S') for r in results],
            'avg_T': [r['avg_T'] for r in results], 'avg_H': [r['avg_H'] for r in results],
            'avg_Soil': [r['avg_Soil'] for r in results], 'total_Rain': [r['total_Rain'] for r in results],
        }
        return jsonify(formatted_results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---
# NEW: Secure Google Weather API Proxy
# ---
@app.route('/weather_forecast')
def get_weather_forecast():
    # Check if the API key was even loaded from the environment
    if not GOOGLE_API_KEY:
        return jsonify({"error": "Server is missing Google API key"}), 500
    
    # These are now secure on the server
    VELLORE_LAT = "12.9165"
    VELLORE_LON = "79.1325"
    GOOGLE_WEATHER_URL = f"https://weather.googleapis.com/v1/forecast:lookup?key={GOOGLE_API_KEY}"
    VELLORE_LOCATION_PAYLOAD = {
        "location": {
            "latitude": float(VELLORE_LAT),
            "longitude": float(VELLORE_LON)
        },
        "params": ["hourlyForecast"],
        "language": "en"
    }

    try:
        # Make the secure, server-to-server request to Google
        response = requests.post(GOOGLE_WEATHER_URL, json=VELLORE_LOCATION_PAYLOAD, timeout=10)
        
        # Check for a bad response from Google
        response.raise_for_status() # This will raise an error for 4xx or 5xx status

        # Success! Pass Google's data directly to our client
        return jsonify(response.json())

    except requests.exceptions.HTTPError as e:
        # Handle API errors from Google (like invalid key)
        return jsonify({"error": f"Google API error: {e.response.text}"}), e.response.status_code
    except requests.exceptions.RequestException as e:
        # Handle network errors
        return jsonify({"error": f"Request to Google Weather failed: {e}"}), 500

@app.route('/pending_events')
def get_pending_events():
    try:
        data_cursor = collection.find(
            {"Risk": {"$ne": "None"}, "ImpactTime": None}
        ).sort("timestamp", DESCENDING)
        events = []
        for doc in data_cursor:
            doc["_id"] = str(doc["_id"]); doc["timestamp"] = doc["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            events.append(doc)
        return jsonify(events)
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/update_impact_time', methods=['POST'])
def update_impact_time():
    try:
        data = request.json
        doc_id, ground_truth_time = data.get('doc_id'), float(data.get('ground_truth_time'))
        if not doc_id or ground_truth_time < 0: return jsonify({"error": "Invalid data"}), 400
        result = collection.update_one({'_id': ObjectId(doc_id)}, {'$set': {'ImpactTime': ground_truth_time}})
        if result.matched_count == 0: return jsonify({"error": "Event not found"}), 404
        print(f"GROUND TRUTH SAVED: Event {doc_id} updated with ImpactTime = {ground_truth_time}h")
        return jsonify({"status": "success", "updated_id": doc_id})
    except Exception as e: return jsonify({"error": str(e)}), 500


# ---
# THIS IS THE NEW, CORRECTED NORMALIZATION FUNCTION
# ---
def normalize_data(sensors_raw):
    sensors_normalized = sensors_raw.copy()
    
    # 1. Normalize Soil
    raw_soil = sensors_raw['Soil']
    if raw_soil > 700: raw_soil = 700
    if raw_soil < 0: raw_soil = 0
    soil_percent = (raw_soil / 700) * 100
    sensors_normalized['Soil'] = soil_percent

    # 2. Normalize Rain
    raw_rain = sensors_raw['Rain']
    if raw_rain > 50:
        sensors_normalized['Rain'] = 1.0
    else: # Otherwise, it's dry.
        sensors_normalized['Rain'] = 0.0
        
    print(f"Data Normalized: Soil {sensors_raw['Soil']}->{soil_percent:.1f}%, Rain {sensors_raw['Rain']}->{sensors_normalized['Rain']}")
    return sensors_normalized


@app.route('/submit_data', methods=['POST'])
def submit_data():
    global previous_risk_state
    
    # 1. Authenticate
    auth_key = request.headers.get('X-API-KEY')
    if auth_key != API_SECRET_KEY:
        print(f"Invalid API key received. {auth_key}")
        return jsonify({"error": "Unauthorized"}), 401
    
    # 2. Get data
    data = request.json
    try:
        sensors_raw = {
            "T": float(data.get('T', 0)), "H": float(data.get('H', 0)), "Soil": float(data.get('Soil', 0)),
            "Rain": float(data.get('Rain', 0)), "Flame": float(data.get('Flame', 0)),
            "Vib": float(data.get('Vib', 0)), "P": float(data.get('P', 0)), "Alt": float(data.get('Alt', 0))
        }
    except Exception as e:
        print(f"Error parsing submitted data: {e}")
        return jsonify({"error": "Bad data format"}), 400

    # 3. NORMALIZE THE DATA
    try:
        sensors_normalized = normalize_data(sensors_raw)
    except Exception as e:
        print(f"Error normalizing data: {e}")
        sensors_normalized = sensors_raw # Fallback

    # 4. Make Prediction (using normalized data)
    try:
        X_features_df = pd.DataFrame([sensors_normalized], columns=FEATURES)
    except KeyError as e:
        print(f"Error: Missing feature {e} from sensor data")
        return jsonify({"error": f"Missing feature {e}"}), 400

    risk_label = "None"
    confidence = 1.0
    predicted_impact_time = 0.0

    if not all([model, time_model, scaler, le]):
        print("Models not loaded, prediction skipped.")
    else:
        with model_lock:
            X_scaled = scaler.transform(X_features_df)
            probabilities = model.predict_proba(X_scaled)[0]
            max_prob_index = np.argmax(probabilities)
            confidence = float(probabilities[max_prob_index]) 
            risk_label = le.inverse_transform([max_prob_index])[0]

            if risk_label.lower() != "none":
                predicted_impact_time = float(round(time_model.predict(X_scaled)[0], 2))
                if predicted_impact_time < 0:
                    predicted_impact_time = 0.0

    ts = datetime.now()
    impact_to_save = 0.0 if risk_label.lower() == "none" else predicted_impact_time

    # 5. Save to MongoDB (We save the *normalized* data)
    mongo_doc = sensors_normalized.copy()
    mongo_doc.update({
        "timestamp": ts, 
        "Risk": risk_label,
        "Confidence": confidence,
        "ImpactTime": impact_to_save,
        "RawSoil": sensors_raw['Soil'], # Optional: save the raw value too
        "RawRain": sensors_raw['Rain']  # Optional: save the raw value too
    })
    collection.insert_one(mongo_doc)

    # 6. Send Notifications
    if risk_label != previous_risk_state and risk_label.lower() != "none":
        print(f"Risk state changed: {previous_risk_state} -> {risk_label}.")
        alert_msg = (
            f"🚨 {risk_label.upper()} Risk Detected ({confidence*100:.0f}% Conf.)\n"
            f"Est. Impact: ~{predicted_impact_time}h\n"
            f"T:{sensors_normalized['T']}°C, H:{sensors_normalized['H']}%"
        )
        threading.Thread(target=send_alert, args=(alert_msg,), daemon=True).start()
    
    previous_risk_state = risk_label
    print(f"{ts.strftime('%Y-%m-%d %H:%M:%S')} | API Data Received | Risk={risk_label} ({confidence*100:.0f}%) | Predicted Impact={predicted_impact_time}h")

    return jsonify({"status": "OK", "predicted_risk": risk_label})

# --- Main Execution ---
if __name__ == '__main__':
    load_models() 
    # REMOVED: threading.Thread(target=fetch_weather_alerts, daemon=True).start()
    print("\n" + "="*50)
    print("   Disaster Prediction Server (v-Cloud) LIVE - LOCAL TEST")
    print(f"  Dashboard running at: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
else:
    load_models()
    # REMOVED: threading.Thread(target=fetch_weather_alerts, daemon=True).start()