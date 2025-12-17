import mne
import matplotlib.pyplot as plt
import numpy as np
import os

# Define the file path for one subject
subject_id = 'sc4002e0'
base_path = 'sleep-edf-database-1.0.0'
edf_file_path = os.path.join(base_path, f'{subject_id}.edf')
hyp_file_path = os.path.join(base_path, f'{subject_id}.hyp')

# 1. Load the raw recording data
raw = mne.io.read_raw_edf(edf_file_path, preload=True, exclude=['EOG horizontal'])
print("Data loaded successfully!")

# 2. Check for annotations in the EDF file first
if len(raw.annotations.description) > 0:
    # This block handles cases where annotations are correctly embedded in the EDF
    print("Found annotations embedded in the EDF file.")
    # (Existing logic for T-labels, etc.)
else:
    # 3. If no annotations in EDF, fall back to the .hyp file
    print(f"No annotations in EDF. Attempting to load: {hyp_file_path}")
    
    # --- ADDING DEFENSIVE CHECKS ---
    if not os.path.exists(hyp_file_path):
        print(f"ERROR: Annotation file not found at {hyp_file_path}")
    else:
        with open(hyp_file_path, 'r') as f:
            lines = f.readlines()
        
        if not lines:
            print(f"ERROR: Annotation file {hyp_file_path} is empty.")
        else:
            try:
                # Let's see what we're actually reading from the file
                descriptions_raw = [line.strip() for line in lines if line.strip()]
                print(f"Found {len(descriptions_raw)} annotations in .hyp file.")
                print(f"First 10 raw descriptions: {descriptions_raw[:10]}")

                descriptions_clean = [d[0] for d in descriptions_raw]
                
                onsets = np.arange(len(descriptions_clean)) * 30
                durations = np.full(len(descriptions_clean), 30)
                
                manual_annotations = mne.Annotations(
                    onset=onsets,
                    duration=durations,
                    description=descriptions_clean,
                    orig_time=raw.info['meas_date']
                )
                raw.set_annotations(manual_annotations)
                
                hyp_map = {'W':'Wake','1':'N1','2':'N2','3':'N3','4':'N3','R':'REM','M':'Movement','?':'Unknown'}
                
                # Make a copy to avoid modifying in place while iterating
                new_desc = list(raw.annotations.description)
                for i, desc in enumerate(new_desc):
                    if desc in hyp_map:
                        new_desc[i] = hyp_map[desc]
                
                # Update the annotations with the mapped descriptions
                raw.annotations.description = np.array(new_desc)

                print("Successfully loaded and mapped annotations from .hyp file.")

            except Exception as e:
                print(f"An unexpected error occurred: {e}")

# 4. Visualize the data
print("Generating plot...")
raw.plot(scalings='auto', duration=300)
plt.show()