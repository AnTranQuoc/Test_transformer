import cv2
import mediapipe as mp
import numpy as np
import os
import re

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

SAVE_DIR = "gesture_data_tokens"
GESTURE_NAME = input("Nhập tên cử chỉ (VD: ok, peace, fist...): ").strip().lower()

if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

existing_files = [f for f in os.listdir(SAVE_DIR) if f.startswith(f"{GESTURE_NAME}_") and f.endswith(".npy")]
existing_ids = [
    int(re.findall(rf"{GESTURE_NAME}_(\d+).npy", f)[0])
    for f in existing_files if re.findall(rf"{GESTURE_NAME}_(\d+).npy", f)
]
sample_id = max(existing_ids) + 1 if existing_ids else 0

print(f"🟡 Bắt đầu thu thập cử chỉ '{GESTURE_NAME}' từ ID {sample_id}")
print("📸 Ấn [s] để lưu mẫu, [q] để thoát.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, f"Gesture: {GESTURE_NAME} | Saved: {sample_id}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Collect Gesture", frame)

    key = cv2.waitKey(1)
    if key == ord('s') and results.multi_hand_landmarks:
        landmarks = results.multi_hand_landmarks[0].landmark
        data = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float32)
        filename = os.path.join(SAVE_DIR, f"{GESTURE_NAME}_{sample_id}.npy")
        np.save(filename, data)
        print(f"✅ Đã lưu: {filename}")
        sample_id += 1
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
