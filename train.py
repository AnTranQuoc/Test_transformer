# train_vit_gesture_classifier.py
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


# --- Dataset ---
class GestureDataset(Dataset):
    def __init__(self, data_dir):
        self.data = []
        self.labels = []
        self.label_map = {}
        files = os.listdir(data_dir)
        for f in files:
            if f.endswith(".npy"):
                label = f.split("_")[0]
                if label not in self.label_map:
                    self.label_map[label] = len(self.label_map)
                path = os.path.join(data_dir, f)
                self.data.append(np.load(path))
                self.labels.append(self.label_map[label])
        # 🔧 Fix: Convert list of ndarray -> numpy array -> tensor (faster)
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(self.labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

# --- Transformer Classifier ---
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=63, num_classes=3, hidden_dim=96, nhead=3, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x shape: (B, 63)
        x = self.embedding(x).unsqueeze(1)  # (B, 1, hidden)
        x = self.transformer(x)             # (B, 1, hidden)
        return self.classifier(x[:, 0])     # (B, num_classes)

# --- Train ---
def train():
    dataset = GestureDataset("gesture_data")
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)
    train_data = torch.utils.data.Subset(dataset, train_idx)
    val_data = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleTransformer(num_classes=len(dataset.label_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(20):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1} - Train Loss: {total_loss / len(train_loader):.4f}")

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                preds = model(x).argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        acc = correct / total * 100
        print(f"✅ Validation Accuracy: {acc:.2f}%")

    torch.save(model.state_dict(), "gesture_transformer.pth")
    print("🎉 Đã lưu mô hình: gesture_transformer.pth")

if __name__ == "__main__":
    train()
