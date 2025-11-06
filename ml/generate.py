import pandas as pd
import numpy as np
import os

N_SAMPLES = 1000
OUT_FILE = "server/sensor_data.csv"

# Ensure the server directory exists
os.makedirs("server", exist_ok=True)

data = []

print(f"Generating {N_SAMPLES} simulated sensor readings...")

for _ in range(N_SAMPLES):
    # Pick a random disaster type for this row
    # We'll make 'None' slightly more common
    dtype = np.random.choice(
        ['Fire', 'Flood', 'Earthquake', 'None'], 
        p=[0.20, 0.20, 0.15, 0.45]
    )

    if dtype == 'Fire':
        row = {
            "T": np.random.normal(loc=90, scale=15),    # High temp
            "H": np.random.normal(loc=15, scale=5),     # Low humidity
            "Soil": np.random.normal(loc=20, scale=10), # Dry soil
            "Rain": 0,                                  # No rain
            "Flame": 1,                                 # Flame detected!
            "Vib": np.random.uniform(0, 0.1),         # Low vibration
            "P": np.random.normal(loc=101, scale=1),    # Normal pressure
            "Alt": np.random.normal(loc=150, scale=10), # Stable altitude
            "Risk": "Fire",
            "ImpactTime": np.random.uniform(0.1, 3.0) # Impacts quickly (hours)
        }
    
    elif dtype == 'Flood':
        row = {
            "T": np.random.normal(loc=18, scale=5),     # Cool temp
            "H": np.random.normal(loc=95, scale=3),     # High humidity
            "Soil": np.random.normal(loc=90, scale=5),  # Saturated soil
            "Rain": 1,                                  # Raining
            "Flame": 0,                                 # No fire
            "Vib": np.random.uniform(0, 0.1),         # Low vibration
            "P": np.random.normal(loc=98, scale=1),     # Low pressure (storm)
            "Alt": np.random.normal(loc=150, scale=10), # Stable altitude
            "Risk": "Flood",
            "ImpactTime": np.random.uniform(2.0, 24.0) # Slower impact (hours)
        }

    elif dtype == 'Earthquake':
        row = {
            "T": np.random.normal(loc=25, scale=8),     # Normal temp
            "H": np.random.normal(loc=50, scale=15),    # Normal humidity
            "Soil": np.random.normal(loc=40, scale=10), # Normal soil
            "Rain": np.random.choice([0, 1], p=[0.9, 0.1]), # Usually no rain
            "Flame": 0,                                 # No fire
            "Vib": np.random.uniform(1.5, 10.0),        # HIGH vibration
            "P": np.random.normal(loc=101, scale=1),    # Normal pressure
            "Alt": np.random.normal(loc=150, scale=10), # Stable altitude
            "Risk": "Earthquake",
            "ImpactTime": np.random.uniform(0.01, 0.5) # Very fast impact (minutes/secs)
        }
    
    else: # 'None'
        row = {
            "T": np.random.normal(loc=25, scale=8),     # Normal temp
            "H": np.random.normal(loc=50, scale=15),    # Normal humidity
            "Soil": np.random.normal(loc=40, scale=10), # Normal soil
            "Rain": 0,                                  # No rain
            "Flame": 0,                                 # No fire
            "Vib": np.random.uniform(0, 0.1),         # Low vibration
            "P": np.random.normal(loc=101, scale=1),    # Normal pressure
            "Alt": np.random.normal(loc=150, scale=10), # Stable altitude
            "Risk": "None",
            "ImpactTime": 0.0                          # No impact
        }
    
    data.append(row)

# Create DataFrame
df = pd.DataFrame(data)

# Ensure columns are in the right order
required = ["T","H","Soil","Rain","Flame","Vib","P","Alt","Risk","ImpactTime"]
df = df[required]

# Save to CSV
df.to_csv(OUT_FILE, index=False)

print(f"Successfully generated {len(df)} samples and saved to {OUT_FILE}")
print("\nData Head:")
print(df.head())