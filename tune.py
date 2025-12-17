import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# --- Load the CORRECT 2D FEATURE Data ---
print("Loading pre-calculated feature data...")
DATA_DIR = 'processed_data'
X_features = np.load(os.path.join(DATA_DIR, 'X_features.npy')) # Load the new file
y = np.load(os.path.join(DATA_DIR, 'y_labels.npy'))
print("Data loaded.")
print(f"Shape of feature data (X): {X_features.shape}") # This will now be 2D

param_grid = {
    'n_estimators': [200, 300],
    'max_depth': [20, None],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(class_weight='balanced', random_state=42, n_jobs=-1)

print("Starting GridSearchCV...")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

grid_search.fit(X_features, y)

print("\nGrid search complete!")
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score (accuracy): ", grid_search.best_score_)