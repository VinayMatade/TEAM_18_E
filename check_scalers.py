#!/usr/bin/env python3
"""
Quick diagnostic script to check scalers.save file
"""
import os
import joblib
import pickle
import numpy as np

SCALER_PATH = "scalers.save"

print("=" * 60)
print("SCALER FILE DIAGNOSTICS")
print("=" * 60)

# 1. Check if file exists
if not os.path.exists(SCALER_PATH):
    print(f"❌ File not found: {SCALER_PATH}")
    exit(1)

# 2. Check file size
size = os.path.getsize(SCALER_PATH)
print(f"✅ File exists: {SCALER_PATH}")
print(f"   Size: {size:,} bytes ({size/1024:.2f} KB)")

# 3. Try joblib
print("\n--- Attempting joblib.load ---")
try:
    obj = joblib.load(SCALER_PATH)
    print(f"✅ joblib.load succeeded")
    print(f"   Type: {type(obj)}")
    
    if isinstance(obj, dict):
        print(f"   Keys: {list(obj.keys())}")
        for k, v in obj.items():
            print(f"   {k}: {type(v)}")
            if hasattr(v, 'mean_'):
                print(f"      mean shape: {v.mean_.shape}")
                print(f"      var shape: {v.var_.shape}")
    elif hasattr(obj, 'mean_'):
        print(f"   Single scaler object")
        print(f"      mean shape: {obj.mean_.shape}")
        print(f"      var shape: {obj.var_.shape}")
    else:
        print(f"   Unexpected object type: {type(obj)}")
        
except Exception as e:
    print(f"❌ joblib.load failed: {e}")

# 4. Try pickle
print("\n--- Attempting pickle.load ---")
try:
    with open(SCALER_PATH, 'rb') as f:
        obj = pickle.load(f)
    print(f"✅ pickle.load succeeded")
    print(f"   Type: {type(obj)}")
    
    if isinstance(obj, dict):
        print(f"   Keys: {list(obj.keys())}")
except Exception as e:
    print(f"❌ pickle.load failed: {e}")

# 5. Try numpy
print("\n--- Attempting np.load ---")
try:
    arr = np.load(SCALER_PATH, allow_pickle=True)
    print(f"✅ np.load succeeded")
    print(f"   Type: {type(arr)}")
    print(f"   Shape: {arr.shape if hasattr(arr, 'shape') else 'N/A'}")
    print(f"   Dtype: {arr.dtype if hasattr(arr, 'dtype') else 'N/A'}")
except Exception as e:
    print(f"❌ np.load failed: {e}")

print("\n" + "=" * 60)
print("RECOMMENDATION:")
print("=" * 60)

if os.path.exists(SCALER_PATH):
    print("If all load methods failed, regenerate scalers.save by running:")
    print("  python train.py")
    print("\nOr use the robust loader in validation.py which will")
    print("automatically recompute scalers from your CSV files.")
else:
    print("Run train.py first to generate scalers.save")

print("=" * 60)
