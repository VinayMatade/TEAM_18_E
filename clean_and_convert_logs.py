#!/usr/bin/env python3
import os
import glob
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
INPUT_FOLDER = "files/logs/"      # Folder with your raw .log files
OUTPUT_FOLDER = "files/cleaned/"  # Folder to save the CSVs

# --- PARAMETER MAPPING ---
# NOTE: All scaling (like Lat/Lng * 1e-7 or HDop * 0.01) is REMOVED
# because your provided text logs are already converted to float degrees.
PARAMS = {
    # GPS (FMT 94): TimeUS,I,Status,GMS,GWk,NSats,HDop,Lat,Lng,Alt,Spd,GCrs...
    'GPS': {
        'indices': {0: 'TimeUS', 5: 'NSats', 6: 'HDop',
                    7: 'Lat', 8: 'Lng', 9: 'Alt',
                    10: 'Spd', 11: 'GCrs'},
        'scalers': {}  # Removed scaling to preserve full precision.
    },
    # GPA (FMT 95): TimeUS,I,VDop,HAcc,VAcc,SAcc...
    'GPA': {
        'indices': {0: 'TimeUS', 3: 'HAcc', 4: 'VAcc', 5: 'SAcc'},
        'scalers': {}
    },
    # IMU (FMT 163): TimeUS,I,GyrX,GyrY,GyrZ,AccX,AccY,AccZ...
    'IMU': {
        'indices': {0: 'TimeUS', 2: 'GyrX', 3: 'GyrY', 4: 'GyrZ',
                    5: 'AccX', 6: 'AccY', 7: 'AccZ'},
        'scalers': {}
    },
    # BARO (FMT 64): TimeUS,I,Alt,AltAMSL...
    'BARO': {
        'indices': {0: 'TimeUS', 2: 'Baro_Alt'},  # Use index 2 for Baro Alt
        'scalers': {}
    },
    # VIBE (FMT 166): TimeUS,IMU,VibeX,VibeY,VibeZ,Clip
    'VIBE': {
        'indices': {0: 'TimeUS', 2: 'VibeX', 3: 'VibeY', 4: 'VibeZ', 5: 'Clip'},
        'scalers': {}
    },
    # ATT (FMT 72): TimeUS,DesRoll,Roll,DesPitch,Pitch,DesYaw,Yaw,AEKF
    'ATT': {
        'indices': {0: 'TimeUS', 2: 'Roll', 4: 'Pitch', 6: 'Yaw'},
        'scalers': {}
    }
}

# --- END CONFIGURATION ---


def parse_and_clean(filepath):
    """Parses a raw log file using specific indices and handles conversions."""
    buffers = {k: [] for k in PARAMS.keys()}

    print(f"Processing {os.path.basename(filepath)}...")

    try:
        with open(filepath, 'r', errors='ignore') as f:
            for line in f:
                # Skip header/FMT lines (which start with FMT, MULT, UNIT, PARM)
                if line.startswith(('FMT', 'MULT', 'UNIT', 'PARM', 'MSG', 'CMD', 'TERR')):
                    continue

                parts = [p.strip() for p in line.split(',')]
                if not parts:
                    continue

                msg_type = parts[0]

                if msg_type in PARAMS:
                    config = PARAMS[msg_type]
                    target_indices = config['indices']
                    scalers = config['scalers']

                    max_idx = max(target_indices.keys())
                    # Check if line has enough columns (parts[0] is MsgName, so parts[1] is index 0)
                    if len(parts) - 1 <= max_idx:
                        continue

                    row_dict = {}
                    try:
                        for idx, name in target_indices.items():
                            # idx is 0-based index AFTER the message type name
                            val = float(parts[idx + 1])

                            # Scaling is skipped/ignored because the scalers dict is empty
                            # if name in scalers: val = val * scalers[name]

                            row_dict[name] = val

                        buffers[msg_type].append(row_dict)
                    except ValueError:
                        continue

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return None

    # Convert buffers to DataFrames and merge
    dfs = []
    for msg_type, rows in buffers.items():
        if rows:
            df = pd.DataFrame(rows)
            df = df.sort_values('TimeUS').drop_duplicates('TimeUS')
            df['TimeUS'] = pd.to_datetime(df['TimeUS'], unit='us')
            df = df.set_index('TimeUS')
            # Rename columns, but exclude generic "Alt" column from renaming if it's from GPS
            # We rename Baro_Alt to prevent collision with GPS_Alt
            rename_map = {}
            for col in df.columns:
                if col == 'Baro_Alt':
                    rename_map[col] = f"BARO_Alt"
                else:
                    rename_map[col] = f"{msg_type}_{col}"
            df = df.rename(columns=rename_map)
            dfs.append(df)

    if not dfs:
        return None

    # Merge strategy: Resample to 50Hz (20ms) and Interpolate
    print(f"   Merging {len(dfs)} message types...")
    full_df = pd.concat(dfs, axis=1)
    full_df = full_df.sort_index()

    # Resample to 50Hz and Interpolate
    resampled = full_df.resample('20ms').mean()
    resampled = resampled.interpolate(method='linear', limit_direction='both')

    return resampled


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # The log files are actually .log, not .csv, so search for both
    log_files = sorted(glob.glob(os.path.join(
        INPUT_FOLDER, "*.log")) + glob.glob(os.path.join(INPUT_FOLDER, "*.txt")))

    # Also check the explicitly uploaded file
    uploaded_file = "00000001.log"
    if os.path.exists(uploaded_file) and uploaded_file not in [os.path.basename(f) for f in log_files]:
        log_files.append(uploaded_file)

    if not log_files:
        print(
            f"No log files found in {INPUT_FOLDER} or the current directory.")
        return

    for f in log_files:
        if not os.path.isfile(f):
            continue

        clean_df = parse_and_clean(f)

        if clean_df is not None and len(clean_df) > 500:
            out_name = os.path.splitext(os.path.basename(f))[
                0] + "_cleaned.csv"
            out_path = os.path.join(OUTPUT_FOLDER, out_name)

            # Save without index label to keep CSV clean
            clean_df.to_csv(out_path, index_label='Time_s')
            print(f"✅ Saved cleaned: {out_name} ({len(clean_df)} rows)")
        else:
            print(
                f"⚠️ Skipping {os.path.basename(f)} (Empty or too short after cleaning/merging).")


if __name__ == "__main__":
    main()
