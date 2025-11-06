from flask import Flask, render_template, jsonify, request
import os, pickle, threading, subprocess, json
import pandas as pd
from datetime import datetime
from pymongo import MongoClient, DESCENDING
from notifier import send_alert
import numpy as np

# --- System Constants for Render ---
# FIX 1: Paths updated for Free Tier (models are in the repo, not a disk)
ML_DIR = "ml" 
MODEL_DIR = os.path.join(ML_DIR, "ml_models")

# --- All auto-retraining code is disabled for the read-only free tier ---
# LAST_TRAIN_COUNT_FILE = os.path.join(ML_DIR, "last_training_count.txt")
# RETRAIN_TRIGGER_COUNT = 500

# --- Locks ---
model_lock = threading.Lock()
# retraining_lock = threading.Lock() # Disabled

# --- Secrets from Render Environment Variables ---
# FIX 2: This is the MOST IMPORTANT FIX.
# You must use the KEY (the name) from your Render dashboard.
# The value (the long password string) STAYS on Render's dashboard ONLY.
MONGODB_URI = os.environ.get("MONGODB_ATLAS_URI") 
API_SECRET_KEY = os.environ.get("API_SECRET_KEY")

if not MONGODB_URI or not API_SECRET_KEY:
    print("FATAL ERROR: MONGODB_ATLAS_URI or API_SECRET_KEY not set in environment.")
    # This error means you forgot to set the variables in the Render "Environment" tab
    # OR you have a typo (like MONGODB_ATLAS_URI vs MONGODB_ATLAS_URI)
    
# --- MongoDB Atlas ---
try:
    # FIX 3: Check if MONGODB_URI is None before trying to connect
    if MONGODB_URI is None:
        raise ValueError("MONGODB_ATLAS_URI environment variable not found.")
        
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.server_info() # Test connection
    db = client["DisasterDB"]
    collection = db["SensorData"]
    print("Connected to MongoDB Atlas.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    # This will now print the "FATAL ERROR" message from above if the key is missing

# --- Flask setup ---
app = Flask(__name__)

# --- ML Model Globals ---
model = None
time_model = None
scaler = None
le = None

def load_models():
    """Loads all 4 models from the Git repo. Thread-safe."""
    global model, time_model, scaler, le, model_lock
    
    # FIX 4: COMMENTED OUT. This line causes a PermissionError on Render's free tier.
    # os.makedirs(MODEL_DIR, exist_ok=True) 
    
    try:
        with model_lock:
            print(f"Attempting to load models from {MODEL_DIR}...")
            model_path = os.path.join(MODEL_DIR, "disaster_model.pkl")
            time_path = os.path.join(MODEL_DIR, "time_predictor.pkl")
            scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
            le_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
            
            # Check if all files exist before trying to load
            for p in [model_path, time_path, scaler_path, le_path]:
                if not os.path.exists(p):
                    print(f"Warning: Model file not found: {p}")
                    print("This means your model is not in your 'ml/ml_models' folder in Git.")
                    return # Exit if models aren't trained yet
            
            model = pickle.load(open(model_path, "rb"))
            time_model = pickle.load(open(time_path, "rb"))
            scaler = pickle.load(open(scaler_path, "rb"))
            le = pickle.load(open(le_path, "rb"))
            print("Models loaded successfully.")
    except Exception as e:
        print(f"An error occurred loading models: {e}")

# --- All auto-retraining functions are disabled for the free tier ---
def get_last_train_count():
    return 0

def set_last_train_count(count):
    # This function is now disabled to prevent PermissionError
    pass 

def _run_training_process(current_row_count):
    # This function is now disabled
    print("[Auto-Retrain] Auto-retraining is disabled on the free tier.")
    pass

def trigger_retraining(current_row_count):
    # This function is now disabled
    pass

# --- Flask Routes ---

@app.route('/')
def dashboard():
    """Serves the main dashboard page."""
    return render_template('dashboard.html')

@app.route('/data')
def get_data():
    """API endpoint for the dashboard to fetch chart data."""
    try:
        # Get last 20 records, sorted newest-first
        data_cursor = collection.find(
            {}, {"_id": 0} # Exclude the _id field
        ).sort("timestamp", DESCENDING).limit(20)
        
        data = list(data_cursor)
        data.reverse() # Data must be chronological (oldest-first)
        
        # Convert datetime objects to strings for JSON
        for item in data:
            item['timestamp'] = item['timestamp'].strftime("%Y-%m-%d %H:%M:%S")
            
        return jsonify(data)
        
    except Exception as e:
        print(f"Error fetching data from MongoDB: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/submit_data', methods=['POST'])
def submit_data():
    """
    NEW API ENDPOINT
    This is where your local_client.py will send its data.
    """
    # 1. Authenticate the request
    auth_key = request.headers.get('X-API-KEY')
    if auth_key != API_SECRET_KEY:
        print(f"Invalid API key received. {auth_key}")
        return jsonify({"error": "Unauthorized"}), 401
    
    # 2. Get the data from the client
    data = request.json
    
    try:
        t, h, soil, rain, flame, vib, p, alt = map(float, [
            data.get('T', 0), data.get('H', 0), data.get('Soil', 0),
            data.get('Rain', 0), data.get('Flame', 0),
            data.get('Vib', 0), data.get('P', 0), data.get('Alt', 0)
        ])
    except Exception as e:
        print(f"Error parsing submitted data: {e}")
        return jsonify({"error": "Bad data format"}), 400

    # 3. Make Prediction
    
    # --- THIS IS THE FIX ---
    # The scaler was trained on a DataFrame with feature names.
    # We must create one here so it knows which column is which.
    
    # First, create a dictionary of the data
    data_dict = {
        "T": t, "H": h, "Soil": soil, "Rain": rain,
        "Flame": flame, "Vib": vib, "P": p, "Alt": alt
    }
    
    # Next, create the DataFrame. The column order MUST match your training script.
    # This list comes from your 'predict.py' file.
    FEATURES = ["T", "H", "Soil", "Rain", "Flame", "Vib", "P", "Alt"]
    X_features_df = pd.DataFrame([data_dict], columns=FEATURES)
    # --- END OF FIX ---

    risk_label = "None"
    impact_time = 0.0

    # Check if models are loaded before predicting
    if not all([model, time_model, scaler, le]):
        print("Models not loaded yet. Skipping prediction.")
    else:
        with model_lock: # Thread-safe prediction
            # Now we scale the DataFrame, not the raw array
            X_scaled = scaler.transform(X_features_df) 
            
            risk_encoded = model.predict(X_scaled)[0]
            risk_label = le.inverse_transform([risk_encoded])[0]
            # And convert from np.float32 to a normal float for MongoDB
            impact_time = float(round(time_model.predict(X_scaled)[0], 2))

    if risk_label.lower() == "none" or impact_time < 0:
        impact_time = 0.0
    
    ts = datetime.now()

    # 4. Save to MongoDB
    mongo_doc = {
        "timestamp": ts, "T": t, "H": h, "Soil": soil, "Rain": rain,
        "Flame": flame, "Vib": vib, "P": p, "Alt": alt,
        "Risk": risk_label, "ImpactTime": impact_time
    }
    collection.insert_one(mongo_doc)

    # 5. Send Notifications (if risk is high)
    if risk_label.lower() not in ["none", ""]: 
        alert_msg = (
            f"🚨 {risk_label.upper()} Risk Detected!\n"
            f"Est. Impact: ~{impact_time}h\n"
            f"T:{t}°C, H:{h}%, Soil:{soil}%, Rain:{rain}, Vib:{vib}"
        )
        threading.Thread(target=send_alert, args=(alert_msg,), daemon=True).start()

    print(f"{ts.strftime('%Y-%m-%d %H:%M:%S')} | API Data Received | Risk={risk_label} | Impact={impact_time}h")

    # FIX 5: Auto-retraining is disabled for the read-only free tier.
    # 6. Check for Auto-Retraining
    # current_row_count = collection.count_documents({})
    # last_train_count = get_last_train_count()
    
    # if (current_row_count - last_train_count) > RETRAIN_TRIGGER_COUNT:
    #     trigger_retraining(current_row_count)
    #     set_last_train_count(current_row_count)

    return jsonify({"status": "OK", "predicted_risk": risk_label}), 200

# --- Main Execution ---
if __name__ == '__main__':
    # This block runs when you test locally (e.g., `python server/app.py`)
    # Gunicorn (Render's server) will NOT run this block.
    
    # Set dummy env vars for local testing if not present
    if not MONGODB_URI: os.environ['MONGODB_ATLAS_URI'] = "YOUR_MONGODB_ATLAS_URI"
    if not API_SECRET_KEY: os.environ['API_SECRET_KEY'] = "test-key"
    
    print("--- RUNNING IN LOCAL DEBUG MODE ---")
    print(f"API Key for local testing: {os.environ['API_SECRET_KEY']}")
    
    load_models() 
    
    print("\nFlask server starting at http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)
else:
    # This block runs WHEN DEPLOYED ON RENDER (using Gunicorn)
    # Load models on startup
    load_models()