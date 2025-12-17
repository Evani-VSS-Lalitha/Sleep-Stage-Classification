import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# -----------------------------
# Load data
# -----------------------------
X = np.load("processed_data_multimodal/X_multimodal_data.npy")   # shape: (N, C, T)
y = np.load("processed_data_multimodal/y_multimodal_labels.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

train_ds = TensorDataset(X_train, y_train)
test_ds = TensorDataset(X_test, y_test)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=32)

# -----------------------------
# CNN Model
# -----------------------------
class SleepCNN(nn.Module):
    def __init__(self):
        super(SleepCNN, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=X.shape[1], out_channels=64, kernel_size=5)
        self.bn1 = nn.BatchNorm1d(64)

        self.conv2 = nn.Conv1d(64, 128, 5)
        self.bn2 = nn.BatchNorm1d(128)

        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(128 * ((X.shape[2] - 8) // 2), 128)
        self.fc2 = nn.Linear(128, 5)  # 5 sleep stages

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)

model = SleepCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -----------------------------
# Train
# -----------------------------
epochs = 10
for epoch in range(epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.4f}")

# -----------------------------
# Evaluate
# -----------------------------
model.eval()
y_pred = []
y_true = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        preds = model(batch_x)
        y_pred.extend(torch.argmax(preds, axis=1).numpy())
        y_true.extend(batch_y.numpy())

print("Accuracy:", accuracy_score(y_true, y_pred))
print(classification_report(y_true, y_pred))
torch.save(model.state_dict(), "cnn_sleep_model.pth")
