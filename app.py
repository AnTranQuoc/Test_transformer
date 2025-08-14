import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import mediapipe as mp
import cv2
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import os

# ---------------- Dataset ----------------
class GestureTokenDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.samples, self.labels = [], []
        self.label_map = {}
        files = sorted(os.listdir(data_dir))
        for f in files:
            if not f.endswith(".npy"):
                continue
            label = f.split("_")[0]
            if label not in self.label_map:
                self.label_map[label] = len(self.label_map)
            arr = np.load(os.path.join(data_dir, f))
            if arr.ndim == 1 and arr.size == 63:
                arr = arr.reshape(21, 3)
            elif arr.ndim == 2 and arr.shape == (21, 3):
                pass
            else:
                continue
            self.samples.append(arr.astype(np.float32))
            self.labels.append(self.label_map[label])
        self.samples = np.array(self.samples)
        self.labels = np.array(self.labels, dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

# ---------------- Positional Encoding ----------------
def sinusoidal_positional_encoding(seq_len, d_model, device=None):
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    position = np.arange(0, seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    pe = torch.from_numpy(pe).unsqueeze(0)
    if device is not None:
        pe = pe.to(device)
    return pe

# ---------------- Model ----------------
class KeypointTransformer(nn.Module):
    def __init__(self, seq_len=21, feat_dim=3, hidden_dim=96, nhead=3, num_layers=2, num_classes=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Linear(feat_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead,
                                                   batch_first=True, dropout=dropout)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        self.register_buffer("pos_encoding", sinusoidal_positional_encoding(seq_len, hidden_dim))

    def forward(self, x):
        x = self.embedding(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        pooled = x.mean(dim=1)
        logits = self.classifier(pooled)
        return logits

# ---------------- Load model ----------------
@st.cache_resource
def load_model(model_path="gesture_transformer_tokens.pth", data_dir="gesture_data_tokens", device="cpu"):
    checkpoint = torch.load(model_path, map_location=device)
    label_map = checkpoint["label_map"]
    num_classes = len(label_map)

    model = KeypointTransformer(seq_len=21, feat_dim=3, hidden_dim=96, nhead=3,
                                 num_layers=2, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    id2label = {v: k for k, v in label_map.items()}

    # Evaluate metrics on validation set
    dataset = GestureTokenDataset(data_dir)
    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=0.2, stratify=dataset.labels, random_state=42)
    val_data = Subset(dataset, val_idx)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(device)
            logits = model(xb)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    report = classification_report(all_labels, all_preds, target_names=[id2label[i] for i in range(num_classes)], output_dict=True)

    return model, id2label, report

# ---------------- MediaPipe ----------------
mp_hands = mp.solutions.hands

# ---------------- UI ----------------
st.set_page_config(page_title="Gesture Transformer Live Demo", layout="wide")
st.title("✋ Gesture Recognition Transformer - Live Camera")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.write(f"**Thiết bị chạy model:** {'GPU' if torch.cuda.is_available() else 'CPU'}")

model, id2label, metrics_report = load_model(device=device)

# Hiển thị metrics
st.sidebar.subheader("📊 Model Metrics (Validation)")
for cls, m in metrics_report.items():
    if isinstance(m, dict):
        st.sidebar.write(f"**{cls}** - Precision: {m['precision']:.2f}, Recall: {m['recall']:.2f}, F1: {m['f1-score']:.2f}")

run_cam = st.checkbox("Bật camera", value=False)
FRAME_WINDOW = st.image([])

if run_cam:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while run_cam:
            ret, frame = cap.read()
            if not ret:
                st.error("Không lấy được khung hình từ camera!")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    keypoints = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                    keypoints = np.array(keypoints, dtype=np.float32)

                    x = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1)
                        conf, pred_id = torch.max(probs, dim=1)
                        pred_label = id2label[pred_id.item()]
                        confidence = conf.item()

                    mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    cv2.putText(frame, f"{pred_label} ({confidence*100:.1f}%)",
                                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    cap.release()
