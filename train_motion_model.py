# =============================================
# train_model.py
# Trains BOTH sign_model.pkl and motion_model.pkl
# Run: python train_model.py
# =============================================

import os
import pandas as pd
import joblib
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR   = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

STATIC_FEATURES = 63      # 21 landmarks × 3
MOTION_FEATURES = 1890    # 30 frames × 63

# =============================================
# TRAIN STATIC MODEL  (A-Y)
# =============================================
print("\n" + "="*50)
print("TRAINING STATIC MODEL (A-Y)")
print("="*50)

X_static, y_static = [], []

for fname in sorted(os.listdir(DATASET_DIR)):
    if not fname.endswith(".csv"):
        continue
    if fname.endswith("_motion.csv"):
        continue

    label = os.path.splitext(fname)[0].upper()
    path  = os.path.join(DATASET_DIR, fname)

    try:
        data = pd.read_csv(path, header=None)
        for _, row in data.iterrows():
            vals = row.values
            if len(vals) == STATIC_FEATURES:
                X_static.append(vals.astype(np.float32))
                y_static.append(label)
            else:
                print(f"  SKIP row in {fname}: expected {STATIC_FEATURES} features, got {len(vals)}")
    except Exception as e:
        print(f"  ERROR reading {fname}: {e}")

print(f"\nSigns found : {sorted(set(y_static))}")
print(f"Total samples: {len(X_static)}")

if len(set(y_static)) < 2:
    print("ERROR: Need at least 2 different signs to train.")
    exit()

# Check minimum samples per class
from collections import Counter
counts = Counter(y_static)
low = [s for s, c in counts.items() if c < 10]
if low:
    print(f"WARNING: These signs have very few samples: {low}")
    print("Collect more data for better accuracy.")

X_tr, X_te, y_tr, y_te = train_test_split(
    X_static, y_static,
    test_size=0.2, random_state=42, stratify=y_static
)

static_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    min_samples_split=2,
    random_state=42,
    n_jobs=-1
)
static_model.fit(X_tr, y_tr)

acc = accuracy_score(y_te, static_model.predict(X_te))
print(f"\nStatic model accuracy: {acc*100:.2f}%")
print("\nPer-sign report:")
print(classification_report(y_te, static_model.predict(X_te)))

static_path = os.path.join(MODEL_DIR, "sign_model.pkl")
joblib.dump(static_model, static_path)
print(f"Static model saved: {static_path}")


# =============================================
# TRAIN MOTION MODEL  (J, Z)
# =============================================
print("\n" + "="*50)
print("TRAINING MOTION MODEL (J, Z)")
print("="*50)

X_motion, y_motion = [], []

for fname in sorted(os.listdir(DATASET_DIR)):
    if not fname.endswith("_motion.csv"):
        continue

    label = fname.replace("_motion.csv", "").upper()
    path  = os.path.join(DATASET_DIR, fname)

    try:
        data = pd.read_csv(path, header=None)
        for _, row in data.iterrows():
            vals = row.values
            if len(vals) == MOTION_FEATURES:
                X_motion.append(vals.astype(np.float32))
                y_motion.append(label)
            else:
                print(f"  SKIP row in {fname}: expected {MOTION_FEATURES} features, got {len(vals)}")
    except Exception as e:
        print(f"  ERROR reading {fname}: {e}")

print(f"\nMotion signs found: {sorted(set(y_motion))}")
print(f"Total sequences   : {len(X_motion)}")

if len(set(y_motion)) < 2:
    print("WARNING: Need both J and Z data for motion model.")
    print("Skipping motion model training.")
else:
    X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
        X_motion, y_motion,
        test_size=0.2, random_state=42, stratify=y_motion
    )

    motion_model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1
    )
    motion_model.fit(X_tr2, y_tr2)

    acc2 = accuracy_score(y_te2, motion_model.predict(X_te2))
    print(f"\nMotion model accuracy: {acc2*100:.2f}%")

    motion_path = os.path.join(MODEL_DIR, "motion_model.pkl")
    joblib.dump(motion_model, motion_path)
    print(f"Motion model saved: {motion_path}")

print("\n" + "="*50)
print("Training complete! Run:  streamlit run app.py")
print("="*50)