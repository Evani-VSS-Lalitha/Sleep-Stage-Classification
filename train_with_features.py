import numpy as np
import os
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load the Pre-processed Data ---
DATA_DIR = 'processed_data'
X_raw = np.load(os.path.join(DATA_DIR, 'X_data.npy'))
y = np.load(os.path.join(DATA_DIR, 'y_labels.npy'))
print("Raw epoch data loaded successfully!")
print(f"Shape of X_raw: {X_raw.shape}")

# --- 2. Advanced Feature Engineering: Relative Band Power ---
BANDS = {'delta': [0.5, 4], 'theta': [4, 8], 'alpha': [8, 12], 'beta': [12, 30]}
SAMPLING_RATE = 100

def calculate_relative_band_power(epoch_data):
    """Calculates the RELATIVE power for each band in a single epoch."""
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

print("Starting feature engineering with relative power...")
features = [calculate_relative_band_power(epoch) for epoch in X_raw]
X_features = np.array(features)

print("Feature engineering complete!")
print(f"Shape of new feature array (X_features): {X_features.shape}")

# --- 3. Train a Random Forest Model ---
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y)
print("\nData prepared for training!")

model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1)

print("Starting model training...")
model.fit(X_train, y_train)
print("Model training complete!")

# --- NEW: Evaluate the Model on the TRAINING Data ---
print("\nEvaluating model performance on the training set...")
y_train_pred = model.predict(X_train)
train_accuracy = np.mean(y_train_pred == y_train)
print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
# ---------------------------------------------------

# --- 4. Evaluate the Model on the TESTING Data ---
print("\nEvaluating model performance on the test set...")
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

print("Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('Actual Label')
plt.xlabel('Predicted Label')
plt.show()