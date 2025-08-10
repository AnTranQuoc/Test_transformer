import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

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

class KeypointTransformer(nn.Module):
    def __init__(self, seq_len=21, feat_dim=3, hidden_dim=96, nhead=3, num_layers=2, num_classes=3, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load("gesture_transformer_tokens.pth", map_location=device)

label_map = checkpoint["label_map"]
inv_label_map = {v: k for k, v in label_map.items()}

model = KeypointTransformer(num_classes=len(label_map)).to(device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

# --- MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

print("🔍 Đang chạy trên:", device)
if device.type == 'cuda':
    print("🚀 GPU:", torch.cuda.get_device_name(0))

cap = cv2.VideoCapture(0)
print("🎬 Nhấn Q để thoát.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    gesture = "..."
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark], dtype=np.float32)
        keypoints = torch.tensor(keypoints).unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(keypoints)
            pred = torch.argmax(output, dim=1).item()
            gesture = inv_label_map.get(pred, "?")

    cv2.putText(frame, f"Gesture: {gesture}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Gesture Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
