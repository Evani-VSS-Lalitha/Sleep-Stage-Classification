import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
import joblib

# Load data
X = np.load("X_final_features.npy")       # shape: (N, features)
y = np.load("y.npy")                      # labels

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# High-accuracy tuned XGBoost
model = XGBClassifier(
    objective="multi:softmax",
    num_class=5,
    eval_metric="mlogloss",
    n_estimators=600,
    max_depth=8,
    learning_rate=0.03,
    subsample=0.9,
    colsample_bytree=0.9,
    gamma=1,
    min_child_weight=3,
    reg_lambda=1.0,
    reg_alpha=0.1,
    tree_method="hist"
)

print("Training XGBoost...")
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print("\n🔥 BEST ML MODEL: XGBoost Accuracy:", acc)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save
joblib.dump(model, "best_xgboost_sleep.pkl")
print("\nModel saved as best_xgboost_sleep.pkl")
