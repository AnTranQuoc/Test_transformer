# realtime_predict.py
import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn

# --- MODEL ---
class SimpleTransformer(nn.Module):
    def __init__(self, input_dim=63, num_classes=3, hidden_dim=96, nhead=3, num_layers=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        return self.classifier(x[:, 0])

# --- LABEL MAP (phải khớp với khi train) ---
label_map = {
    0: "fist",
    1: "ok",
    2: "open_hand"
}

# --- Load Model ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = SimpleTransformer(num_classes=3).to(device)
model.load_state_dict(torch.load("gesture_transformer.pth", map_location=device))
model.eval()

# --- MediaPipe ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# --- Webcam ---
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

        keypoints = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]).flatten()
        keypoints = torch.tensor(keypoints, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            output = model(keypoints)
            pred = torch.argmax(output, dim=1).item()
            gesture = label_map.get(pred, "?")

    # Hiển thị cử chỉ dự đoán
    cv2.putText(frame, f"Gesture: {gesture}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
    cv2.imshow("Gesture Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
