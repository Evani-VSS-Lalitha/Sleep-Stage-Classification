import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


# ===========================================
# Load Data
# ===========================================
X = np.load("X_multimodal_features.npy")    # shape: (N, channels, timesteps)
y = np.load("y.npy")

X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.long)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
test_loader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)


# ===========================================
# Attention Layer
# ===========================================
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        weights = torch.softmax(self.attn(lstm_output), dim=1)
        context = torch.sum(weights * lstm_output, dim=1)
        return context


# ===========================================
# CNN + BiLSTM + Attention Model
# ===========================================
class SleepModel(nn.Module):
    def __init__(self, channels, timesteps, num_classes=5):
        super().__init__()

        # CNN feature extractor
        self.conv1 = nn.Conv1d(channels, 64, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.relu = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)

        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        # Attention
        self.attn = Attention(256)

        # Final classifier
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # x: (B, C, T)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))

        # reshape for LSTM -> (B, T, features)
        x = x.permute(0, 2, 1)

        # BiLSTM output
        lstm_out, _ = self.lstm(x)

        # Attention
        context = self.attn(lstm_out)

        # Final prediction
        return self.fc(context)


model = SleepModel(
    channels=X.shape[1],
    timesteps=X.shape[2],
    num_classes=5
)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# ===========================================
# Training Loop
# ===========================================
epochs = 15
for epoch in range(epochs):
    model.train()
    running_loss = 0

    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_x)
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {running_loss/len(train_loader):.4f}")


# ===========================================
# Evaluation
# ===========================================
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        preds = model(batch_x)
        y_pred.extend(torch.argmax(preds, dim=1).numpy())
        y_true.extend(batch_y.numpy())

print("\nBest DL Model Accuracy:", accuracy_score(y_true, y_pred))
print("\nClassification Report:\n", classification_report(y_true, y_pred))

torch.save(model.state_dict(), "best_dl_sleep_model.pth")
print("\nModel saved as best_dl_sleep_model.pth")
