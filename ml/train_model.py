import os, sys, pickle, argparse
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, mean_squared_error
from pymongo import MongoClient

# --- Setup Argument Parser ---
# This lets Render's `app.py` tell this script where to save models
# The default is Render's standard persistent disk mount path.
parser = argparse.ArgumentParser(description="Disaster Model Training Script")
parser.add_argument('--model-dir', type=str, default="/app/ml", 
                    help="Directory to save models and count file (e.g., /app/ml on Render)")
args = parser.parse_args()

# Use the path from the command line argument
ML_DIR = args.model_dir
OUTDIR = os.path.join(ML_DIR, "ml_models")
os.makedirs(OUTDIR, exist_ok=True)

# --- MongoDB Connection ---
# Reads the connection string from Render's Environment Variables
MONGODB_URI = os.environ.get("MONGODB_ATLAS_URI")
if not MONGODB_URI:
    print("FATAL ERROR: MONGODB_ATLAS_URI not set in environment.")
    sys.exit(1)

try:
    client = MongoClient(MONGODB_URI, serverSelectionTimeoutMS=5000)
    client.server_info()
    db = client["DisasterDB"]
    collection = db["SensorData"]
except Exception as e:
    print(f"Training script failed to connect to MongoDB: {e}")
    sys.exit(1)

print("Starting training script...")
print(f"Model save directory: {OUTDIR}")
print("Reading data from MongoDB...")

# --- Load Data from Mongo ---
try:
    # Find all documents, exclude _id and timestamp
    cursor = collection.find({}, {"_id": 0, "timestamp": 0})
    df = pd.DataFrame(list(cursor))
    
    if df.empty:
        print("No data found in MongoDB. Exiting training (this is ok if DB is new).")
        sys.exit(0) # Exit successfully
        
    print(f"Successfully loaded {len(df)} rows from database.")

except Exception as e:
    print(f"Error loading data from MongoDB: {e}")
    sys.exit(1)


# --- Feature Engineering ---
required = ["T","H","Soil","Rain","Flame","Vib","P","Alt","Risk","ImpactTime"]
for c in required:
    if c not in df.columns:
        print(f"Missing required column {c} in database. Exiting.")
        sys.exit(1)

df = df.dropna(subset=required)
if len(df) < 100:
    print(f"Not enough data to train (found {len(df)} rows). Need at least 100.")
    sys.exit(0) # Exit successfully, not an error

X = df[["T","H","Soil","Rain","Flame","Vib","P","Alt"]].astype(float)
y_labels = df["Risk"]
targ = df["ImpactTime"].astype(float)

print("Data prepared. Starting model training...")

# --- Label Encoding ---
le = LabelEncoder()
y_encoded = le.fit_transform(y_labels)

# --- Standard Scaling ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- 1. XGBoost Classifier ---
clf = xgb.XGBClassifier(
    objective='multi:softmax', 
    num_class=len(le.classes_), 
    random_state=42, 
    use_label_encoder=False,
    eval_metric='mlogloss'
)
clf.fit(X_scaled, y_encoded)
print("Classifier training complete.")

# --- 2. XGBoost Regressor ---
reg = xgb.XGBRegressor(
    objective='reg:squarederror',
    random_state=42,
    eval_metric='rmse'
)
reg.fit(X_scaled, targ)
print("Regressor training complete.")

# --- Save All 4 Models ---
try:
    pickle.dump(clf, open(os.path.join(OUTDIR, "disaster_model.pkl"), "wb"))
    pickle.dump(reg, open(os.path.join(OUTDIR, "time_predictor.pkl"), "wb"))
    pickle.dump(scaler, open(os.path.join(OUTDIR, "scaler.pkl"), "wb"))
    pickle.dump(le, open(os.path.join(OUTDIR, "label_encoder.pkl"), "wb"))
    print(f"All 4 models saved to {OUTDIR}")
except Exception as e:
    print(f"Error saving models: {e}")
    sys.exit(1)

# --- Final Report ---
y_pred_encoded = clf.predict(X_scaled)
y_pred_labels = le.inverse_transform(y_pred_encoded)

print("\n--- Training Complete: Final Report ---")
print(f"Data rows used: {len(df)}")
print(f"Classes found: {list(le.classes_)}")
print("\nClassification Report (on training data):")
print(classification_report(y_labels, y_pred_labels))
print("\nRegression MSE (on training data):")
print(mean_squared_error(targ, reg.predict(X_scaled)))
print("--- End of Training Script ---")