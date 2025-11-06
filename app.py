from flask import Flask, render_template, jsonify, request
import os, pickle, threading, subprocess, json
import pandas as pd
from datetime import datetime
from pymongo import MongoClient, DESCENDING
from notifier import send_alert
import numpy as np

# --- System Constants for Render ---
# Render's persistent disk is mounted at /app/ml
ML_DIR = "/app/ml" 
LAST_TRAIN_COUNT_FILE = os.path.join(ML_DIR, "last_training_count.txt")
MODEL_DIR = os.path.join(ML_DIR, "ml_models")
RETRAIN_TRIGGER_COUNT = 500  # Retrain after 500 new rows

# --- Locks ---
model_lock = threading.Lock()       # Prevents reading models while they are being reloaded
retraining_lock = threading.Lock()  # Prevents starting a new training if one is already running

# --- Secrets from Render Environment Variables ---
MONGODB_URI = os.environ.get("mongodb+srv://kannamreddyjaideepreddy_db_user:<db_password>@disasterdb.0wn492l.mongodb.net/?appName=DisasterDB")
API_SECRET_KEY = os.environ.get("API_SECRET_KEY") # This is a password you will create

if not MONGODB_URI or not API_SECRET_KEY:
    print("FATAL ERROR: MONGODB_ATLAS_URI or API_SECRET_KEY not set in environment.")
    # In a real server, you might exit or raise an error
    # For now, we'll just print the warning.
    # exit(1) # Uncomment this line for production

# --- MongoDB Atlas ---
try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.server_info() # Test connection
    db = client["DisasterDB"]
    collection = db["SensorData"]
    print("Connected to MongoDB Atlas.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    # In production, this should be a fatal error
    # exit(1)

# --- Flask setup ---
app = Flask(__name__)

# --- ML Model Globals ---
model = None
time_model = None
scaler = None
le = None

def load_models():
    """Loads all 4 models from the persistent disk. Thread-safe."""
    global model, time_model, scaler, le, model_lock
    
    # Ensure the model directory exists on the persistent disk
    os.makedirs(MODEL_DIR, exist_ok=True) 
    
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
                    print("This is normal on first boot. Run train_model.py from Render Shell.")
                    return # Exit if models aren't trained yet
            
            model = pickle.load(open(model_path, "rb"))
            time_model = pickle.load(open(time_path, "rb"))
            scaler = pickle.load(open(scaler_path, "rb"))
            le = pickle.load(open(le_path, "rb"))
            print("Models loaded successfully.")
    except Exception as e:
        print(f"An error occurred loading models: {e}")

def get_last_train_count():
    """Reads the last training row count from its file."""
    try:
        with open(LAST_TRAIN_COUNT_FILE, 'r') as f:
            return int(f.read())
    except (FileNotFoundError, ValueError):
        return 0

def set_last_train_count(count):
    """Saves the new training row count to its file."""
    try:
        with open(LAST_TRAIN_COUNT_FILE, 'w') as f:
            f.write(str(int(count)))
    except Exception as e:
        print(f"Error writing to {LAST_TRAIN_COUNT_FILE}: {e}")


def _run_training_process(current_row_count):
    """
    Internal function to run the training script and reload models.
    This runs in a separate thread.
    """
    global retraining_lock
    
    if retraining_lock.locked():
        print("[Auto-Retrain] A training process is already running. Skipping.")
        return

    with retraining_lock:
        try:
            print(f"[Auto-Retrain] Triggered at {current_row_count} rows. Starting training...")
            
            # Run the training script in a separate process
            # We must tell it where the persistent disk is
            process = subprocess.run(
                ['python', 'ml/train_model.py', f'--model-dir={ML_DIR}'], 
                capture_output=True, text=True, check=True
            )
            print("[Auto-Retrain] Subprocess output:", process.stdout)
            
            print("[Auto-Retrain] Training complete. Hot-reloading models...")
            load_models() # This function is thread-safe
            
            set_last_train_count(current_row_count)
            print(f"[Auto-Retrain] Success. New baseline count: {current_row_count}")
            
        except subprocess.CalledProcessError as e:
            print(f"[Auto-Retrain] ERROR: Training script failed.")
            print("STDOUT:", e.stdout)
            print("STDERR:", e.stderr)
        except Exception as e:
            print(f"[Auto-Retrain] An unexpected error occurred: {e}")

def trigger_retraining(current_row_count):
    """Starts the retraining process in a new non-blocking thread."""
    print("[Auto-Retrain] Queuing retraining job...")
    train_thread = threading.Thread(
        target=_run_training_process, 
        args=(current_row_count,), 
        daemon=True
    )
    train_thread.start()

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
    X_features = np.array([t, h, soil, rain, flame, vib, p, alt]).reshape(1, -1)
    risk_label = "None"
    impact_time = 0.0

    # Check if models are loaded before predicting
    if not all([model, time_model, scaler, le]):
        print("Models not loaded yet. Skipping prediction.")
    else:
        with model_lock: # Thread-safe prediction
            X_scaled = scaler.transform(X_features)
            risk_encoded = model.predict(X_scaled)[0]
            risk_label = le.inverse_transform([risk_encoded])[0]
            impact_time = round(time_model.predict(X_scaled)[0], 2)

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

    # 6. Check for Auto-Retraining
    # This is a simple way to count, for high performance use Mongo's .count_documents()
    current_row_count = collection.count_documents({})
    last_train_count = get_last_train_count()
    
    if (current_row_count - last_train_count) > RETRAIN_TRIGGER_COUNT:
        trigger_retraining(current_row_count)
        # Update in-memory count immediately to prevent re-triggering
        set_last_train_count(current_row_count)

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