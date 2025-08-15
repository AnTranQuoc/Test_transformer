# Hand Gesture Recognition with TransformerEncoder

Real-time hand gesture recognition from keypoint data extracted using **MediaPipe**, powered by a **TransformerEncoder** model.  
The model can classify the following gestures:

- 👌 OK
- ✊ Fist
- 🖐 Open Hand
- ☝ One Finger
- ✌ Two Fingers
- 🤟 Three Fingers
- 🖖 Four Fingers
- 🖕 Offensive Gesture

---

## 🎥 Video Demo
[![Watch the demo](demo_thumbnail.png)](demo_video.mp4)  
*Click the image above to watch the demo video.*

---

## 📊 Dataset
- **Source:** Keypoint data collected using **MediaPipe Hands**.
- **Format:** Each sample is a `(21, 3)` array containing `(x, y, z)` coordinates of 21 hand landmarks.
- **Preprocessing:**
  - Normalize position (translate wrist to origin).
  - Scale hand size to a consistent ratio.
- **Classes:** 8 gesture classes (OK, Fist, Open Hand, 1, 2, 3, 4, Offensive Gesture).

---

## 🧠 Transformer Overview
The model uses a **TransformerEncoder** to process the sequence of keypoints  
(21 points per hand, each with coordinates `(x, y, z)`).

**Processing Pipeline:**

Keypoints Sequence → Positional Encoding → TransformerEncoder → Pooling → Classifier

1. **MediaPipe** detects the hand and extracts `(21×3)` keypoints.
2. Normalize the keypoints (scale, translate).
3. Each frame → one vector (`sequence length = 21`, `feature dim = 3`).
4. Apply **Positional Encoding** to add spatial position information.
5. **TransformerEncoder** learns spatial relationships between keypoints.
6. **Pooling + Fully Connected Layer + Softmax** → predict gesture label.

---

## 📦 Quick Installation

You can install the dependencies in two ways:

**1️⃣ Using `requirements.txt`:**
```bash
pip install -r requirements.txt
```
**2. Install directly**
```bash
pip install torch mediapipe opencv-python numpy scikit-learn
```
***3. Deploy with Streamlit***
```bash
streamlit run app.py
```
