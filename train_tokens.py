import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

# ---------------- Dataset ----------------
class GestureTokenDataset(Dataset):
    """
    Mỗi sample trả về (seq_len=21, feat_dim=3), label
    Hỗ trợ file .npy có shape (63,) hoặc (21,3).
    Label lấy từ tiền tố filename trước dấu '_'
    """
    def __init__(self, data_dir):
        self.samples = []
        self.labels = []
        self.label_map = {}
        files = sorted(os.listdir(data_dir))
        for f in files:
            if not f.endswith(".npy"):
                continue
            label = f.split("_")[0]
            if label not in self.label_map:
                self.label_map[label] = len(self.label_map)
            arr = np.load(os.path.join(data_dir, f))
            # normalize/reshape
            if arr.ndim == 1 and arr.size == 63:
                arr = arr.reshape(21, 3)
            elif arr.ndim == 2 and arr.shape[0] == 21 and arr.shape[1] == 3:
                pass
            else:
                # try to flatten then reshape, else skip
                try:
                    arr = arr.flatten()[:63].reshape(21, 3)
                except Exception:
                    print(f"Skipping file with unexpected shape: {f}, shape={arr.shape}")
                    continue
            self.samples.append(arr.astype(np.float32))
            self.labels.append(self.label_map[label])

        self.samples = np.array(self.samples)  # (N,21,3)
        self.labels = np.array(self.labels).astype(np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# ---------------- Positional Encoding ----------------
def sinusoidal_positional_encoding(seq_len, d_model, device=None):
    """
    Return tensor shape (1, seq_len, d_model)
    """
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    position = np.arange(0, seq_len)[:, np.newaxis]  # (seq_len,1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = torch.from_numpy(pe).unsqueeze(0)  # (1, seq_len, d_model)
    if device is not None:
        pe = pe.to(device)
    return pe

# ---------------- Transformer Classifier ----------------
class KeypointTransformer(nn.Module):
    def __init__(self, seq_len=21, feat_dim=3, hidden_dim=96, nhead=3, num_layers=2, num_classes=3, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # embed each keypoint (3-d) -> hidden_dim
        self.embedding = nn.Linear(feat_dim, hidden_dim)

        # transformer encoder: set batch_first=True so input shape is (B, seq, embed)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                   batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # classifier head
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )

        # initialize positional embedding (sinusoidal or learnable if you prefer)
        self.register_buffer("pos_encoding", sinusoidal_positional_encoding(seq_len, hidden_dim))

    def forward(self, x):
        # x: (B, seq_len, feat_dim)
        B = x.size(0)
        x = self.embedding(x)                  # (B, seq_len, hidden_dim)
        # add positional encoding (broadcast)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)                # (B, seq_len, hidden_dim)
        # pooling across sequence -> mean pooling
        pooled = x.mean(dim=1)                 # (B, hidden_dim)
        logits = self.classifier(pooled)       # (B, num_classes)
        return logits

# ---------------- Training Loop ----------------
def train(data_dir="gesture_data_tokens", epochs=30, batch_size=32, lr=1e-3, device=None):
    device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    dataset = GestureTokenDataset(data_dir)
    if len(dataset) == 0:
        raise RuntimeError("No samples found in data_dir or shapes were unexpected.")

    num_classes = len(dataset.label_map)
    print(f"Found {len(dataset)} samples, classes: {dataset.label_map}")

    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, random_state=42, stratify=dataset.labels)
    train_data = torch.utils.data.Subset(dataset, train_idx)
    val_data = torch.utils.data.Subset(dataset, val_idx)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    model = KeypointTransformer(seq_len=21, feat_dim=3, hidden_dim=96, nhead=3, num_layers=2, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)            # (B,21,3)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)               # (B, num_classes)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * xb.size(0)
        avg_loss = running_loss / len(train_loader.dataset)
        print(f"Epoch {epoch}/{epochs} - Train loss: {avg_loss:.4f}")

        # validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)
        acc = 100.0 * correct / total if total > 0 else 0.0
        print(f"  ✅ Val Accuracy: {acc:.2f}% ({correct}/{total})")

    # save
    torch.save({
        "model_state": model.state_dict(),
        "label_map": dataset.label_map,
    }, "gesture_transformer_tokens.pth")
    print("Saved model -> gesture_transformer_tokens.pth")

if __name__ == "__main__":
    train()