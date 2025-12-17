import numpy as np
import os
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load the RAW Data ---
DATA_DIR = 'processed_data'
X_raw = np.load(os.path.join(DATA_DIR, 'X_data.npy'))
y = np.load(os.path.join(DATA_DIR, 'y_labels.npy'))
print("Raw epoch data loaded successfully!")

# --- 2. Feature Engineering ---
BANDS = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 12], 'beta': [12, 30]}
SAMPLING_RATE = 100

def calculate_relative_band_power(epoch_data):
    freqs, psd = signal.welch(epoch_data, fs=SAMPLING_RATE)
    abs_powers = []
    for band in BANDS.values():
        freq_indices = np.where((freqs >= band[0]) & (freqs < band[1]))[0]
        band_power = np.trapz(psd[:, freq_indices], freqs[freq_indices], axis=1)
        abs_powers.append(band_power)
    abs_powers = np.array(abs_powers).T
    total_power = np.sum(abs_powers, axis=1, keepdims=True)
    relative_powers = np.divide(abs_powers, total_power, out=np.zeros_like(abs_powers), where=total_power!=0)
    return relative_powers.flatten()

print("Starting feature engineering...")
features = [calculate_relative_band_power(epoch) for epoch in X_raw]
X_features = np.array(features)
print("Feature engineering complete!")
print(f"Shape of new feature array (X_features): {X_features.shape}")

# --- NEW STEP: SAVE THE CALCULATED FEATURES ---
np.save(os.path.join(DATA_DIR, 'X_features.npy'), X_features)
print(f"Saved 2D feature array to 'X_features.npy'")
# -----------------------------------------------

# --- 3. Train and Evaluate the Model ---
# (The rest of the script trains the model as before for verification)
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)
print("\nStarting model training...")
model.fit(X_train, y_train)
print("Model training complete!")
# ... (evaluation and plotting code remains the same)