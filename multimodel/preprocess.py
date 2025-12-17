# multimodel/preprocess.py (Final Version)
import mne
import numpy as np
import os
import glob

# --- Configuration ---
DATA_DIR = 'sleep-edf-database-1.0.0'
OUTPUT_DIR = 'processed_data_final' # New folder for final 11-feature data
EPOCH_DURATION_S = 30
EEG_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz']
# Define all required physiological channels
REQUIRED_CHANNELS_BASE = EEG_CHANNELS + ['Resp oro-nasal', 'Temp body']

os.makedirs(OUTPUT_DIR, exist_ok=True)
edf_files = glob.glob(os.path.join(DATA_DIR, '*.edf'))
all_epochs_data, all_epochs_labels = [], []
master_event_id = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}

print(f"Found {len(edf_files)} EDF files.")

for edf_file in edf_files:
    # (Same robust loading and channel finding logic as before)
    # This part is shortened for brevity, but use your full robust script
    subject_id = os.path.basename(edf_file).replace('.edf', '')
    print(f"\n--- Processing subject: {subject_id} ---")
    try:
        raw = mne.io.read_raw_edf(edf_file, preload=True, stim_channel=None, verbose='WARNING')
        available_channels = raw.ch_names
        
        emg_channel_name = None
        if 'EMG Submental' in available_channels: emg_channel_name = 'EMG Submental'
        elif 'EMG submental' in available_channels: emg_channel_name = 'EMG submental'
        
        REQUIRED_CHANNELS = REQUIRED_CHANNELS_BASE + ([emg_channel_name] if emg_channel_name else [])

        if not all(ch in available_channels for ch in REQUIRED_CHANNELS):
            print(f"  - WARNING: Subject missing required channels. Skipping.")
            continue
        
        # (The rest of your robust annotation parsing logic goes here...)
        hyp_file = edf_file.replace('.edf', '.hyp')
        binary_label_map = {0:'Wake',1:'N1',2:'N2',3:'N3',4:'REM',5:'Movement',6:'Unknown'}
        with open(hyp_file, 'rb') as f: data = f.read()
        stages = [binary_label_map[b] for b in data[::2] if b in binary_label_map]
        annots = mne.Annotations(onset=np.arange(len(stages))*30, duration=[30]*len(stages), description=stages, orig_time=raw.info['meas_date'])
        raw.set_annotations(annots)

        raw.pick(REQUIRED_CHANNELS)
        raw.filter(l_freq=0.3, h_freq=35, picks=['eeg'], fir_design='firwin')
        
        events, event_id = mne.events_from_annotations(raw, event_id=master_event_id, chunk_duration=30)
        epochs = mne.Epochs(raw=raw, events=events, event_id=event_id, tmin=0, tmax=30, preload=True, baseline=None)
        
        all_epochs_data.append(epochs.get_data())
        all_epochs_labels.append(epochs.events[:, 2])
        print(f"  - SUCCESS: Created {len(epochs)} epochs with {len(REQUIRED_CHANNELS)} channels.")

    except Exception as e:
        print(f"  - ERROR: {e}")

# Save the final 11-feature raw data
if all_epochs_data:
    X = np.concatenate(all_epochs_data, axis=0)
    y = np.concatenate(all_epochs_labels, axis=0)
    np.save(os.path.join(OUTPUT_DIR, 'X_final_raw_data.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_final_labels.npy'), y)
    print(f"\nFinal pre-processing complete! Data saved to '{OUTPUT_DIR}'.")