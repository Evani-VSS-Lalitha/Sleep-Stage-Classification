import mne
import numpy as np
import os
import glob

# --- Configuration ---
DATA_DIR = 'sleep-edf-database-1.0.0'
OUTPUT_DIR = 'processed_data'
EPOCH_DURATION_S = 30

os.makedirs(OUTPUT_DIR, exist_ok=True)
edf_files = glob.glob(os.path.join(DATA_DIR, '*.edf'))

# --- Main Processing Loop ---
all_epochs_data = []
all_epochs_labels = []

# Global map for all possible stages we care about
master_event_id = {'Wake': 0, 'N1': 1, 'N2': 2, 'N3': 3, 'REM': 4}

for edf_file in edf_files:
    subject_id = os.path.basename(edf_file).replace('.edf', '')
    print(f"\nProcessing subject: {subject_id}...")

    try:
        raw = mne.io.read_raw_edf(edf_file, preload=True, stim_channel=None, verbose='WARNING')
        hyp_file = edf_file.replace('.edf', '.hyp')

        if not os.path.exists(hyp_file):
            print(f"  - WARNING: Annotation file not found for {subject_id}. Skipping.")
            continue

        # --- FIX #1: CORRECTLY PARSE BINARY DATA (SKIP NULL BYTES) ---
        binary_label_map = {0: 'Wake', 1: 'N1', 2: 'N2', 3: 'N3', 4: 'REM', 5: 'Movement', 6: 'Unknown'}
        with open(hyp_file, 'rb') as f:
            data = f.read()
        
        # Read every OTHER byte by slicing the data array [::2]
        stages_from_bytes = data[::2]
        
        mapped_descriptions = [binary_label_map[byte_val] for byte_val in stages_from_bytes if byte_val in binary_label_map]
        
        if not mapped_descriptions:
            print(f"  - WARNING: No valid sleep stages found in {hyp_file}. Skipping.")
            continue

        onsets = np.arange(len(mapped_descriptions)) * EPOCH_DURATION_S
        durations = np.full(len(mapped_descriptions), EPOCH_DURATION_S)
        annotations = mne.Annotations(onset=onsets, duration=durations, description=mapped_descriptions, orig_time=raw.info['meas_date'])
        raw.set_annotations(annotations)
        print(f"  - Successfully parsed {len(mapped_descriptions)} stages from binary .hyp file.")

        raw.pick(['EEG Fpz-Cz', 'EEG Pz-Oz'])
        raw.filter(l_freq=0.3, h_freq=35, fir_design='firwin', skip_by_annotation='edge')
        print(f"  - Data filtered.")

        # --- FIX #3: CREATE EVENT_ID DYNAMICALLY ---
        # Find which stages are present in this specific recording
        present_stages = set(raw.annotations.description)
        # Create an event_id dict only for the stages we have and care about
        subject_event_id = {stage: master_event_id[stage] for stage in present_stages if stage in master_event_id}
        
        if not subject_event_id:
            print("  - WARNING: No relevant sleep stages found for this subject. Skipping.")
            continue

        events, _ = mne.events_from_annotations(raw, event_id=subject_event_id, chunk_duration=EPOCH_DURATION_S)
        
        # --- FIX #2: DISABLE BASELINE CORRECTION ---
        epochs = mne.Epochs(raw=raw, events=events, event_id=subject_event_id, 
                            tmin=0, tmax=EPOCH_DURATION_S, preload=True, 
                            baseline=None, verbose='WARNING') # Add baseline=None
        print(f"  - Created {len(epochs)} epochs.")

        all_epochs_data.append(epochs.get_data())
        all_epochs_labels.append(epochs.events[:, 2])

    except Exception as e:
        print(f"  - ERROR processing {subject_id}: {e}")

# --- Final Step: Combine and Save ---
if not all_epochs_data:
    print("\n----------------------------------------------------")
    print("ERROR: No subjects were processed successfully. No data was saved.")
    print("Please check the error messages above.")
    print("----------------------------------------------------")
else:
    X = np.concatenate(all_epochs_data, axis=0)
    y = np.concatenate(all_epochs_labels, axis=0)
    np.save(os.path.join(OUTPUT_DIR, 'X_data.npy'), X)
    np.save(os.path.join(OUTPUT_DIR, 'y_labels.npy'), y)
    print("\n----------------------------------------------------")
    print("Pre-processing complete!")
    print(f"Shape of final data array (X): {X.shape}")
    print(f"Shape of final labels array (y): {y.shape}")
    print(f"Data saved to '{OUTPUT_DIR}' directory.")
    print("----------------------------------------------------")
print(X.shape)