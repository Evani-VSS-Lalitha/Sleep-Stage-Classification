import os
import glob

# The path to your dataset folder
data_folder = 'sleep-edf-database-1.0.0'

# Find all files ending with .rec in the folder
rec_files = glob.glob(os.path.join(data_folder, '*.rec'))

# Loop through the list of .rec files and rename them
for old_path in rec_files:
    # Create the new path by replacing .rec with .edf
    new_path = old_path.replace('.rec', '.edf')
    
    # Rename the file
    os.rename(old_path, new_path)
    print(f"Renamed '{old_path}' to '{new_path}'")

print("\nRenaming complete!")