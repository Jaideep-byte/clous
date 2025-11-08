from flask import Flask, render_template, jsonify, request
import serial, csv, os, pickle, threading, subprocess, json, time, requests
import pandas as pd
from datetime import datetime, timedelta
from pymongo import MongoClient, DESCENDING
from notifier import send_alert
import numpy as np
from bson.objectid import ObjectId

# --- Constants ---
RETRAIN_TRIGGER_COUNT = 500
ML_DIR = "ml"
LAST_TRAIN_COUNT_FILE = os.path.join(ML_DIR, "last_training_count.txt")
MODEL_DIR = os.path.join(ML_DIR, "ml_models")
SERIAL_PORT = "COM7"
BAUD_RATE = 9600

# --- FIX: Ensure this order matches exactly the training script ---
FEATURES = ["T", "H", "Soil", "Rain", "Flame", "Vib", "P", "Alt"]

# --- Weather API Config (Open-Meteo - Free) ---
VELLORE_LAT, VELLORE_LON = "12.9165", "79.1325"
WEATHER_API_URL = (
    f"https://api.open-meteo.com/v1/forecast?latitude={VELLORE_LAT}&longitude={VELLORE_LON}"
    "&hourly=precipitation_probability&weather_alerts=auto&timezone=auto&name=Vellore"
)

# --- Locks ---
model_lock = threading.Lock()
retraining_lock = threading.Lock()

# --- MongoDB Setup ---
MONGODB_URI = os.environ.get(
    "MONGODB_URI",
    "mongodb://127.0.0.1:27017/?directConnection=true&serverSelectionTimeoutMS=2000&appName=mongosh+2.5.9"
)
try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    db = client["DisasterDB"]
    collection = db["SensorData"]
    print("✅ Connected to MongoDB")
except Exception as e:
    print(f"❌ MongoDB Connection Error: {e}")
    exit()

# --- Flask setup ---
app = Flask(__name__)
CSV_FILE = "server/sensor_data.csv"
CSV_HEADER = ["Timestamp","T","H","Soil","Rain","Flame","Vib","P","Alt","Risk","Confidence","ImpactTime"]
os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        csv.writer(f).writerow(CSV_HEADER)

# --- Global State ---
model = time_model = scaler = le = None
previous_risk_state = "None"
last_weather_alert = {"text": "No official weather alerts for Vellore.", "timestamp": None}

# =====================================================
#                 MODEL HANDLING
# =====================================================
def load_models():
    global model, time_model, scaler, le
    try:
        with model_lock:
            print(f"🔄 Loading models from {MODEL_DIR}...")
            model = pickle.load(open(os.path.join(MODEL_DIR, "disaster_model.pkl"), "rb"))
            time_model = pickle.load(open(os.path.join(MODEL_DIR, "time_predictor.pkl"), "rb"))
            scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
            le = pickle.load(open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb"))
            print("✅ Models loaded successfully.")
    except Exception as e:
        print(f"⚠️ Warning: Failed to load models: {e}")
        model = time_model = scaler = le = None

def get_last_train_count():
    try:
        with open(LAST_TRAIN_COUNT_FILE, "r") as f:
            return int(f.read().strip())
    except:
        return 0

def set_last_train_count(count):
    os.makedirs(ML_DIR, exist_ok=True)
    with open(LAST_TRAIN_COUNT_FILE, "w") as f:
        f.write(str(int(count)))

def _run_training_process(current_row_count):
    if retraining_lock.locked(): return
    with retraining_lock:
        try:
            print(f"[Auto-Retrain] Triggered at {current_row_count} rows.")
            process = subprocess.run(
                ["python", "ml/train_model.py"], capture_output=True, text=True, check=True
            )
            print(process.stdout)
            load_models()
            set_last_train_count(current_row_count)
            print(f"[Auto-Retrain] ✅ Done. New baseline: {current_row_count}")
        except Exception as e:
            print(f"[Auto-Retrain] ❌ Error: {e}")

def trigger_retraining(current_row_count):
    threading.Thread(target=_run_training_process, args=(current_row_count,), daemon=True).start()

# =====================================================
#                 WEATHER ALERT FETCHER
# =====================================================
def fetch_weather_alerts():
    global last_weather_alert
    while True:
        print("🌦️ Fetching weather alerts...")
        try:
            response = requests.get(WEATHER_API_URL, timeout=10)
            response.raise_for_status()
            data = response.json()
            alert_text = "No official weather alerts for Vellore."
            if "weather_alerts" in data and "alerts" in data["weather_alerts"]:
                alerts = data["weather_alerts"]["alerts"]
                if alerts:
                    a = alerts[0]
                    title = a.get("title", "Weather Alert")
                    desc = a.get("description", "No details.")
                    sender = a.get("sender", "Official Source")
                    alert_text = f"**{title.upper()}** (from {sender})\n{desc}"
            last_weather_alert = {"text": alert_text, "timestamp": datetime.now()}
            print(f"✅ Weather alert updated: {alert_text.splitlines()[0]}")
        except Exception as e:
            print(f"⚠️ Weather fetch failed: {e}")
        time.sleep(1800)

# =====================================================
#                 SERIAL READER
# =====================================================
def read_serial():
    global previous_risk_state
    last_train_count = get_last_train_count()
    current_row_count = collection.count_documents({})
    print(f"📡 Serial reader started. Rows: {current_row_count}, Baseline: {last_train_count}")

    ser = None
    while ser is None:
        try:
            ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
            print(f"✅ Connected to {SERIAL_PORT}")
        except serial.SerialException as e:
            print(f"❌ Serial connect failed: {e}. Retrying...")
            time.sleep(5)

    while True:
        try:
            line = ser.readline().decode("utf-8", errors="ignore").strip()
        except Exception as e:
            print(f"⚠️ Serial error: {e}. Reconnecting...")
            time.sleep(5)
            try:
                ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
            except:
                continue
            continue

        if not line or ":" not in line or "|" not in line:
            continue

        try:
            data = dict(item.split(":") for item in line.split("|") if ":" in item)
            sensors = {k: float(data.get(k, 0)) for k in FEATURES}
        except Exception:
            print(f"⚠️ Parse error: {line}")
            continue

        # --- Prediction ---
        risk_label, confidence, predicted_impact_time = "None", 1.0, 0.0

        if model and scaler and le:
            with model_lock:
                X = np.array([sensors[f] for f in FEATURES]).reshape(1, -1)
                X_scaled = scaler.transform(X)
                probs = model.predict_proba(X_scaled)[0]
                idx = np.argmax(probs)
                confidence = float(probs[idx])
                risk_label = le.inverse_transform([idx])[0]

                # Optional confidence threshold (to suppress false Fire/Flood)
                if confidence < 0.6:
                    risk_label = "None"

                if risk_label.lower() != "none":
                    predicted_impact_time = float(max(0, round(time_model.predict(X_scaled)[0], 2)))

        ts = datetime.now()
        ts_str = ts.strftime("%Y-%m-%d %H:%M:%S")

        # Save to CSV
        with open(CSV_FILE, "a", newline="") as f:
            csv.writer(f).writerow(
                [ts_str] + list(sensors.values()) + [risk_label, f"{confidence:.2f}", predicted_impact_time]
            )

        # Save to MongoDB
        mongo_doc = sensors.copy()
        mongo_doc.update({
            "timestamp": ts,
            "Risk": risk_label,
            "Confidence": confidence,
            "ImpactTime": predicted_impact_time,
        })
        collection.insert_one(mongo_doc)
        current_row_count += 1

        # Alert on change
        if risk_label != previous_risk_state and risk_label.lower() != "none":
            msg = (
                f"🚨 {risk_label.upper()} Risk ({confidence*100:.0f}%)\n"
                f"Predicted Impact ≈ {predicted_impact_time}h\n"
                f"T={sensors['T']}°C, H={sensors['H']}%"
            )
            threading.Thread(target=send_alert, args=(msg,), daemon=True).start()

        previous_risk_state = risk_label
        print(f"{ts_str} | Risk={risk_label} ({confidence*100:.0f}%) | Impact={predicted_impact_time}h")

        if (current_row_count - last_train_count) >= RETRAIN_TRIGGER_COUNT:
            trigger_retraining(current_row_count)
            last_train_count = current_row_count

# =====================================================
#                 FLASK ROUTES
# =====================================================
@app.route("/")
def dashboard():
    return render_template("dashboard.html")

@app.route("/data")
def get_data():
    try:
        data = list(collection.find({}, {"_id": 0}).sort("timestamp", DESCENDING).limit(20))
        for d in data:
            d["timestamp"] = d["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
        data.reverse()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/historical_data")
def get_historical_data():
    try:
        hours = 24 * 7 if request.args.get("period") == "7d" else 24
        start_time = datetime.now() - timedelta(hours=hours)
        pipeline = [
            {"$match": {"timestamp": {"$gte": start_time}}},
            {"$group": {
                "_id": {"$dateTrunc": {"date": "$timestamp", "unit": "hour"}},
                "avg_T": {"$avg": "$T"},
                "avg_H": {"$avg": "$H"},
                "avg_Soil": {"$avg": "$Soil"},
                "total_Rain": {"$sum": "$Rain"},
            }},
            {"$sort": {"_id": 1}},
        ]
        results = list(collection.aggregate(pipeline))
        formatted = {
            "labels": [r["_id"].strftime("%Y-%m-%dT%H:%M:%S") for r in results],
            "avg_T": [r["avg_T"] for r in results],
            "avg_H": [r["avg_H"] for r in results],
            "avg_Soil": [r["avg_Soil"] for r in results],
            "total_Rain": [r["total_Rain"] for r in results],
        }
        return jsonify(formatted)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/weather_alert")
def get_weather_alert():
    return jsonify(last_weather_alert)

@app.route("/update_impact_time", methods=["POST"])
def update_impact_time():
    try:
        data = request.json
        doc_id = data.get("doc_id")
        impact = float(data.get("ground_truth_time"))
        if not doc_id or impact < 0:
            return jsonify({"error": "Invalid data"}), 400
        collection.update_one({"_id": ObjectId(doc_id)}, {"$set": {"ImpactTime": impact}})
        print(f"✅ Ground truth updated for {doc_id}: {impact}h")
        return jsonify({"status": "success", "id": doc_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# =====================================================
#                 MAIN ENTRY
# =====================================================
if __name__ == "__main__":
    load_models()
    threading.Thread(target=read_serial, daemon=True).start()
    threading.Thread(target=fetch_weather_alerts, daemon=True).start()
    print("\n" + "="*55)
    print("🌍 Disaster Prediction Server (v3.3 - Stable Local)")
    print("🚀 Dashboard: http://localhost:5000")
    print("="*55 + "\n")
    app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
