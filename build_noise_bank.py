import numpy as np
import pandas as pd
import glob
import os
from scipy.ndimage import gaussian_filter1d

# --- CONFIG ---
# CRITICAL: Point this to your CLEANED folder, not the raw logs
INPUT_FOLDER = "files/cleaned/"
OUTPUT_FILE = "noise_bank.npy"
METERS_PER_DEG = 111139.0

collected_noise = []
files = glob.glob(os.path.join(INPUT_FOLDER, "*.csv"))
print(f"Found {len(files)} clean CSVs to extract noise from...")

for f in files:
    try:
        df = pd.read_csv(f)

        # Check if we have the required columns
        if 'GPS_Lat' not in df.columns or 'GPS_Lng' not in df.columns:
            continue

        lat = df['GPS_Lat'].values
        lon = df['GPS_Lng'].values

        if len(lat) < 500:
            continue

        # 1. Convert to Meters relative to start
        lat_m = (lat - lat[0]) * METERS_PER_DEG
        lon_m = (lon - lon[0]) * METERS_PER_DEG

        # 2. Filter Outliers (Sanity Check)
        # If the drone moved > 5km in one file, it might be a valid long flight,
        # but if the NOISE is huge, we catch it later.

        # 3. Calculate Smooth Path (The "Truth")
        # Sigma=10 at 50Hz is approx 0.2-0.5s smoothing window
        smooth_lat = gaussian_filter1d(lat_m, sigma=10)
        smooth_lon = gaussian_filter1d(lon_m, sigma=10)

        # 4. Extract Vibration (Raw - Smooth)
        noise_lat = lat_m - smooth_lat
        noise_lon = lon_m - smooth_lon

        # 5. Sanity Check for GPS Glitches
        # Real vibration is usually < 2.0 meters.
        # If we see a jump of 50 meters, it's a glitch, not noise. Drop it.
        if np.std(noise_lat) > 5.0 or np.std(noise_lon) > 5.0:
            print(
                f"⚠️ Skipping {os.path.basename(f)}: Noise too high (Glitch detected)")
            continue

        # Stack (N, 2)
        file_noise = np.stack([noise_lat, noise_lon], axis=1)
        collected_noise.append(file_noise)

    except Exception as e:
        print(f"Skipped {os.path.basename(f)}: {e}")

if collected_noise:
    final_bank = np.concatenate(collected_noise, axis=0)
    np.save(OUTPUT_FILE, final_bank)
    print(f"✅ Saved {len(final_bank)} samples to {OUTPUT_FILE}")
    # The Std should be small (e.g., 0.1m to 1.5m)
    print(
        f"   Stats: Mean={np.mean(final_bank):.4f}m, Std={np.std(final_bank):.4f}m")
else:
    print("❌ No valid data found.")
