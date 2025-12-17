import numpy as np
import os
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from tcn import TCN  # Import the TCN layer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
import seaborn as sns
import matplotlib.pyplot as plt

# --- 1. Load the Pre-processed Data (2-channel EEG) ---
DATA_DIR = 'processed_data'
X = np.load(os.path.join(DATA_DIR, 'X_data.npy'))
y = np.load(os.path.join(DATA_DIR, 'y_labels.npy'))
print("Data loaded successfully!")

# --- 2. Calculate Class Weights to Handle Imbalance ---
unique_classes = np.unique(y)
weights = class_weight.compute_class_weight('balanced', classes=unique_classes, y=y)
class_weights_dict = dict(zip(unique_classes, weights))
print(f"Calculated class weights: {class_weights_dict}")

# --- 3. Prepare Data ---
y_reshaped = y.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False)
y_onehot = encoder.fit_transform(y_reshaped)
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42, stratify=y)

# Reshape data for the model: (samples, steps, features)
X_train_reshaped = np.transpose(X_train, (0, 2, 1))
X_test_reshaped = np.transpose(X_test, (0, 2, 1))
print("\nData prepared for training!")

# --- 4. Build the TCN Model Architecture ✨ ---
input_layer = Input(shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2]))

# The TCN layer is powerful at learning temporal patterns
tcn_layer = TCN(nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8, 16], return_sequences=False)(input_layer)
tcn_layer = BatchNormalization()(tcn_layer)
tcn_layer = Dropout(0.5)(tcn_layer)

# Final classification layer
output_layer = Dense(y_train.shape[1], activation='softmax')(tcn_layer)

model = Model(inputs=input_layer, outputs=output_layer)

# --- 5. Compile and Train ---
optimizer = Adam(learning_rate=0.001) # TCNs can often handle a slightly higher learning rate
early_stopping = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

print("\nStarting model training...")
history = model.fit(
    X_train_reshaped, y_train,
    epochs=100, # Train for more epochs, EarlyStopping will find the best one
    batch_size=64,
    validation_data=(X_test_reshaped, y_test),
    callbacks=[early_stopping],
    class_weight=class_weights_dict # Apply the class weights!
)
print("Model training complete!")

# --- 6. Evaluate and Report ---
loss, test_accuracy = model.evaluate(X_test_reshaped, y_test, verbose=0)
print(f"\nFinal Test Accuracy: {test_accuracy * 100:.2f}%")

# (The rest of the plotting and reporting code remains the same)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy vs. Epochs')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss vs. Epochs')
plt.legend()
plt.show()

y_pred = model.predict(X_test_reshaped)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
print("\nClassification Report:")
class_names = ['Wake', 'N1', 'N2', 'N3', 'REM']
print(classification_report(y_test_classes, y_pred_classes, target_names=class_names, zero_division=0))

print("Confusion Matrix:")
cm = confusion_matrix(y_test_classes, y_pred_classes)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
plt.show()