from flask import Flask, render_template, jsonify, request
import os, pickle, threading, subprocess, json, time, requests
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient, DESCENDING
from notifier import send_alert
import numpy as np
from bson.objectid import ObjectId

# --- System Constants ---
ML_DIR = "ml"  # Models are in the 'ml/ml_models' folder in your Git repo
MODEL_DIR = os.path.join(ML_DIR, "ml_models")

# This is the feature order your model was trained on.
FEATURES = ["T", "H", "Soil", "Rain", "Flame", "Vib", "P", "Alt"]

# --- Weather API Config (Open-Meteo - No Key Needed!) ---
VELLORE_LAT = "12.9165"
VELLORE_LON = "79.1325"
WEATHER_API_URL = (
    f"https://api.open-meteo.com/v1/forecast?latitude={VELLORE_LAT}&longitude={VELLORE_LON}"
    "&hourly=precipitation_probability&weather_alerts=auto&timezone=auto&name=Vellore"
)

# --- Locks ---
model_lock = threading.Lock()
# retraining_lock = threading.Lock() # Disabled for free tier

# --- Secrets from Render Environment Variables ---
MONGODB_URI = os.environ.get("MONGODB_ATLAS_URI")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")

if not MONGODB_URI or not API_SECRET_KEY:
    print("FATAL ERROR: MONGODB_ATLAS_URI or API_SECRET_KEY not set in environment.")
    # This error means you forgot to set the variables in the Render "Environment" tab
    
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
    exit() # Must exit if we can't connect to the DB

# --- Flask setup ---
app = Flask(__name__)

# --- Global State ---
model, time_model, scaler, le = None, None, None, None
last_weather_alert = {"text": "No official weather alerts for Vellore.", "timestamp": None}
previous_risk_state = "None" # Used for sending alerts only on state *change*

# --- Model Loading & (Disabled) Training Functions ---
def load_models():
    """Loads all 4 models from the Git repo. Thread-safe."""
    global model, time_model, scaler, le, model_lock
    
    # os.makedirs(MODEL_DIR, exist_ok=True) # <-- DISABLED for Render Free Tier (causes PermissionError)
    
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
        print("Please ensure your .pkl files are in the 'ml/ml_models' folder in GitHub.")

# --- Auto-training is disabled for Render's read-only file system ---
def get_last_train_count():
    return 0
def set_last_train_count(count):
    pass # Disabled
def _run_training_process(current_row_count):
    print("[Auto-Retrain] Auto-retraining is disabled on the free tier.")
    pass # Disabled
def trigger_retraining(current_row_count):
    pass # Disabled

# --- Official Weather Alert Fetcher (from your local code) ---
def fetch_weather_alerts():
    """
    Runs in a background thread to check for official alerts from Open-Meteo.
    Runs every 30 minutes. No API key needed.
    """
    global last_weather_alert, WEATHER_API_URL
    while True:
        print("Fetching new weather alerts for Vellore (from Open-Meteo)...")
        try:
            response = requests.get(WEATHER_API_URL, timeout=10)
            response.raise_for_status() # Raise an error for bad status codes
            data = response.json()
            
            alert_text = "No official weather alerts for Vellore."
            
            if 'weather_alerts' in data and 'alerts' in data['weather_alerts'] and len(data['weather_alerts']['alerts']) > 0:
                alert = data['weather_alerts']['alerts'][0]
                title = alert.get('title', 'Weather Alert')
                sender = alert.get('sender', 'Official Source')
                desc = alert.get('description', 'No details provided.')
                alert_text = f"**{title.upper()}** (from {sender})\n{desc}"
            
            last_weather_alert = {"text": alert_text, "timestamp": datetime.now()}
            print(f"Weather alert updated: {alert_text.splitlines()[0]}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            last_weather_alert = {"text": f"Error fetching weather alerts: {e}", "timestamp": datetime.now()}
        
        # Sleep for 30 minutes before the next check
        time.sleep(1800)

# --- Flask Routes ---
@app.route('/')
def dashboard():
    """Serves the main dashboard page."""
    return render_template('dashboard.html')

@app.route('/data')
def get_data():
    """API endpoint for the dashboard to fetch chart data."""
    try:
        data_cursor = collection.find({}, {"_id": 0}).sort("timestamp", DESCENDING).limit(20)
        data = list(data_cursor); data.reverse()
        for item in data: item['timestamp'] = item['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
        return jsonify(data)
    except Exception as e: return jsonify({"error": str(e)}), 500

@app.route('/historical_data')
def get_historical_data():
    """API endpoint for the historical charts (from your local code)"""
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

@app.route('/weather_alert')
def get_weather_alert():
    """API endpoint for the weather alert box (from your local code)"""
    global last_weather_alert
    return jsonify(last_weather_alert)

@app.route('/pending_events')
def get_pending_events():
    """API endpoint for the human-in-the-loop feedback (from your local code)"""
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
    """API endpoint for saving human feedback (from your local code)"""
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
# THIS IS THE /submit_data ENDPOINT YOUR LOCAL CLIENT TALKS TO.
# IT DOES NOT READ FROM A SERIAL PORT. IT WAITS FOR A POST REQUEST.
# ---
@app.route('/submit_data', methods=['POST'])
def submit_data():
    global previous_risk_state
    
    # This is our canary test.
    print("--- SERVER IS RUNNING THE FINAL, CORRECTED v3.0 CODE ---")
    
    # 1. Authenticate the request
    auth_key = request.headers.get('X-API-KEY')
    if auth_key != API_SECRET_KEY:
        print(f"Invalid API key received. {auth_key}")
        return jsonify({"error": "Unauthorized"}), 401
    
    # 2. Get the data from the client
    data = request.json
    
    try:
        # Create a dictionary of all sensor values
        sensors = {
            "T": float(data.get('T', 0)), "H": float(data.get('H', 0)), "Soil": float(data.get('Soil', 0)),
            "Rain": float(data.get('Rain', 0)), "Flame": float(data.get('Flame', 0)),
            "Vib": float(data.get('Vib', 0)), "P": float(data.get('P', 0)), "Alt": float(data.get('Alt', 0))
        }
    except Exception as e:
        print(f"Error parsing submitted data: {e}")
        return jsonify({"error": "Bad data format"}), 400

    # 3. Make Prediction
    # This logic fixes the bad "Fire" prediction.
    
    # Build Feature DataFrame in the correct order
    try:
        X_features_df = pd.DataFrame([sensors], columns=FEATURES)
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
            # Scale the DataFrame (fixes the UserWarning)
            X_scaled = scaler.transform(X_features_df)
            
            probabilities = model.predict_proba(X_scaled)[0]
            max_prob_index = np.argmax(probabilities)
            
            # Cast to float (fixes the BSON error)
            confidence = float(probabilities[max_prob_index]) 
            risk_label = le.inverse_transform([max_prob_index])[0]

            if risk_label.lower() != "none":
                # Cast to float (fixes the BSON error)
                predicted_impact_time = float(round(time_model.predict(X_scaled)[0], 2))
                if predicted_impact_time < 0:
                    predicted_impact_time = 0.0

    ts = datetime.now()
    impact_to_save = 0.0 if risk_label.lower() == "none" else predicted_impact_time

    # 4. Save to MongoDB
    mongo_doc = sensors.copy()
    mongo_doc.update({
        "timestamp": ts, 
        "Risk": risk_label,
        "Confidence": confidence,
        "ImpactTime": impact_to_save
    })
    collection.insert_one(mongo_doc)

    # 5. Send Notifications
    if risk_label != previous_risk_state and risk_label.lower() != "none":
        print(f"Risk state changed: {previous_risk_state} -> {risk_label}.")
        alert_msg = (
            f"🚨 {risk_label.upper()} Risk Detected ({confidence*100:.0f}% Conf.)\n"
            f"Est. Impact: ~{predicted_impact_time}h\n"
            f"T:{sensors['T']}°C, H:{sensors['H']}%"
        )
        threading.Thread(target=send_alert, args=(alert_msg,), daemon=True).start()
    
    previous_risk_state = risk_label
    print(f"{ts.strftime('%Y-%m-%d %H:%M:%S')} | API Data Received | Risk={risk_label} ({confidence*100:.0f}%) | Predicted Impact={predicted_impact_time}h")

    # 6. Auto-Retraining (Disabled)
    # ... code is removed ...

    return jsonify({"status": "OK", "predicted_risk": risk_label})

# --- Main Execution ---
if __name__ == '__main__':
    # This block runs when you test locally (e.g., `python app.py`)
    # Gunicorn (Render's server) will NOT run this block.
    
    load_models() 
    # Start the NEW weather alert fetcher
    threading.Thread(target=fetch_weather_alerts, daemon=True).start()
    
    print("\n" + "="*50)
    print("   Disaster Prediction Server (v-Cloud) LIVE - LOCAL TEST")
    print(f"  Dashboard running at: http://localhost:5000")
    print("="*50 + "\n")
    app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)
else:
    # This block runs WHEN DEPLOYED ON RENDER (using Gunicorn)
    # Load models on startup
    load_models()
    # Start the weather alert thread on production
    threading.Thread(target=fetch_weather_alerts, daemon=True).start()