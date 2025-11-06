import pickle
import pandas as pd
import numpy as np
import os
import sys

MODEL_DIR = "ml/ml_models"

# --- 1. Load all 4 models ---
try:
    clf = pickle.load(open(os.path.join(MODEL_DIR, "disaster_model.pkl"), "rb"))
    reg = pickle.load(open(os.path.join(MODEL_DIR, "time_predictor.pkl"), "rb"))
    scaler = pickle.load(open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb"))
    le = pickle.load(open(os.path.join(MODEL_DIR, "label_encoder.pkl"), "rb"))
except FileNotFoundError:
    print(f"Error: Model files not found in {MODEL_DIR}.")
    print("Please run train_model.py first.")
    sys.exit(1)

# Define the feature order explicitly (must match training)
FEATURES = ["T", "H", "Soil", "Rain", "Flame", "Vib", "P", "Alt"]

def get_prediction(sensor_data):
    """
    Takes a dictionary of sensor data and returns a prediction.
    """
    try:
        # --- 2. Create DataFrame in the correct order ---
        # This is critical. The model expects columns in the exact training order.
        input_df = pd.DataFrame([sensor_data], columns=FEATURES)
    except ValueError:
        print("Error: Input data dictionary might be malformed.")
        return None

    # --- 3. Scale the data ---
    # Use scaler.transform, NOT scaler.fit_transform
    input_scaled = scaler.transform(input_df)

    # --- 4. Make predictions ---
    
    # Predict Risk (Classification)
    risk_encoded = clf.predict(input_scaled)
    # Decode the numeric label (e.g., 2) back to a string (e.g., 'Fire')
    risk_label = le.inverse_transform(risk_encoded)[0] 
    
    # Predict Impact Time (Regression)
    impact_time = reg.predict(input_scaled)[0]

    # --- 5. Format the result ---
    
    # Don't predict a negative impact time
    if impact_time < 0:
        impact_time = 0.0

    # If risk is "None", impact time doesn't matter
    if risk_label == "None":
        impact_time = 0.0

    return {
        "PredictedRisk": risk_label,
        "EstimatedImpactTime_Hours": round(impact_time, 2)
    }

# --- Main execution block to test the function ---
if __name__ == "__main__":
    
    # Example: Simulating a "Fire" event
    test_data_fire = {
        "T": 95.0,     # High temp
        "H": 12.0,     # Low humidity
        "Soil": 15.0,  # Dry soil
        "Rain": 0,     # No rain
        "Flame": 1,    # Flame detected
        "Vib": 0.1,    # Low vibration
        "P": 101.2,    # Normal pressure
        "Alt": 149.0   # Stable altitude
    }
    
    # Example: Simulating a "Flood" event
    test_data_flood = {
        "T": 17.0,     # Cool temp
        "H": 98.0,     # High humidity
        "Soil": 88.0,  # Saturated soil
        "Rain": 1,     # Raining
        "Flame": 0,    # No fire
        "Vib": 0.0,    # Low vibration
        "P": 98.5,     # Low pressure
        "Alt": 150.0   # Stable altitude
    }
    
    # Example: Simulating a "Normal" day
    test_data_none = {
        "T": 25.0,     # Normal temp
        "H": 55.0,     # Normal humidity
        "Soil": 40.0,  # Normal soil
        "Rain": 0,     # No rain
        "Flame": 0,    # No fire
        "Vib": 0.1,    # Low vibration
        "P": 101.0,    # Normal pressure
        "Alt": 151.0   # Stable altitude
    }

    print("--- Making Test Predictions ---")
    
    print("\nTest Case 1: Fire")
    pred_fire = get_prediction(test_data_fire)
    print(pred_fire)

    print("\nTest Case 2: Flood")
    pred_flood = get_prediction(test_data_flood)
    print(pred_flood)
    
    print("\nTest Case 3: None")
    pred_none = get_prediction(test_data_none)
    print(pred_none)