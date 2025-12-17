# multimodel/train_multimodal_model.py (Final Version)
import numpy as np
import os
from scipy import signal
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Load the new 5-channel raw data
DATA_DIR = 'processed_data_final'
X_raw = np.load(os.path.join(DATA_DIR, 'X_final_raw_data.npy'))
y = np.load(os.path.join(DATA_DIR, 'y_final_labels.npy'))
print(f"Final raw data loaded. Shape: {X_raw.shape}")

# Feature engineering for 5 channels (EEGx2, EMG, Resp, Temp)
def calculate_final_features(epoch_data):
    eeg_data = epoch_data[:2, :]
    emg_data = epoch_data[2, :]
    resp_data = epoch_data[3, :]
    temp_data = epoch_data[4, :]
    
    # EEG Features (8)
    freqs, psd = signal.welch(eeg_data, fs=100)
    BANDS = {'d':[0.5,4],'t':[4,8],'a':[8,12],'b':[12,30]}
    abs_p = np.array([np.trapz(psd[:,(freqs>=b[0])&(freqs<b[1])], freqs[(freqs>=b[0])&(freqs<b[1])]) for b in BANDS.values()]).T
    rel_p = (abs_p / np.sum(abs_p, axis=1, keepdims=True)).flatten()
    
    # EMG, Respiration, Temp Features (3)
    emg_feat = np.log(np.var(emg_data) + 1e-9)
    resp_feat = np.std(resp_data) # Breathing variability
    temp_feat = np.mean(temp_data) # Average body temp
    
    return np.concatenate([rel_p, [emg_feat, resp_feat, temp_feat]])

print("Starting final feature engineering...")
X_features = np.array([calculate_final_features(epoch) for epoch in X_raw])
print(f"Final features created. Shape: {X_features.shape}") # Should be (..., 11)

# Train the final model
X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

print("\n--- Final Model Evaluation ---")
print(classification_report(y_test, model.predict(X_test), target_names=['Wake','N1','N2','N3','REM'], zero_division=0))

# Save the final model and features
joblib.dump(model, 'sleep_final_model.pkl')
np.save('X_final_features.npy', X_features)
print("\nFinal 11-feature model and features saved!")