import numpy as np
import os
import mne
from scipy import signal

# --- This script will process ONLY ONE subject and save their features AND raw data ---

# --- Configuration ---
SUBJECT_ID = 'st7022j0' # You can change this to any subject ID
DATA_DIR = 'sleep-edf-database-1.0.0'

def calculate_final_features(epoch_data):
    # (This function is identical to the one in your training script)
    eeg_data = epoch_data[:2, :]
    emg_data = epoch_data[2, :]
    resp_data = epoch_data[3, :]
    temp_data = epoch_data[4, :]
    BANDS = {'d':[0.5,4],'t':[4,8],'a':[8,12],'b':[12,30]}
    freqs, psd = signal.welch(eeg_data, fs=100)
    abs_p = np.array([np.trapz(psd[:,(freqs>=b[0])&(freqs<b[1])], freqs[(freqs>=b[0])&(freqs<b[1])]) for b in BANDS.values()]).T
    rel_p = (abs_p / np.sum(abs_p, axis=1, keepdims=True)).flatten()
    emg_feat = np.log(np.var(emg_data) + 1e-9)
    resp_feat = np.std(resp_data)
    temp_feat = np.mean(temp_data)
    return np.concatenate([rel_p, [emg_feat, resp_feat, temp_feat]])

# --- Main Logic ---
print(f"Processing single subject: {SUBJECT_ID}")
edf_file = os.path.join(DATA_DIR, f'{SUBJECT_ID}.edf')
hyp_file = os.path.join(DATA_DIR, f'{SUBJECT_ID}.hyp')

raw = mne.io.read_raw_edf(edf_file, preload=True, stim_channel=None, verbose='WARNING')
REQUIRED_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EMG Submental', 'Resp oro-nasal', 'Temp body']
raw.pick(REQUIRED_CHANNELS)

binary_label_map = {0:'Wake',1:'N1',2:'N2',3:'N3',4:'REM',5:'Movement',6:'Unknown'}
with open(hyp_file, 'rb') as f: data = f.read()
stages = [binary_label_map[b] for b in data[::2] if b in binary_label_map]
annots = mne.Annotations(onset=np.arange(len(stages))*30, duration=[30]*len(stages), description=stages, orig_time=raw.info['meas_date'])
raw.set_annotations(annots)

events, event_id = mne.events_from_annotations(raw, event_id={'Wake':0,'N1':1,'N2':2,'N3':3,'REM':4}, chunk_duration=30)
epochs = mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=0, tmax=30, preload=True, baseline=None)
X_raw_single = epochs.get_data()
print(f"Created {len(X_raw_single)} epochs for this subject.")

# Calculate features
X_features_single = np.array([calculate_final_features(epoch) for epoch in X_raw_single])

# --- Save both the feature file AND the raw data file ---
raw_output_filename = f'subject_{SUBJECT_ID}_raw_data.npy'
features_output_filename = f'subject_{SUBJECT_ID}_features.npy'
np.save(raw_output_filename, X_raw_single)
np.save(features_output_filename, X_features_single)

print("\n----------------------------------------------------")
print(f"Success! Files for one subject have been saved:")
print(f"   - Raw Data: '{raw_output_filename}' (for the Epoch Explorer)")
print(f"   - Features: '{features_output_filename}' (for the Full Night Analysis)")
print("----------------------------------------------------")