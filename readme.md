# Hand Gesture Recognition with TransformerEncoder

Nhận diện cử chỉ tay thời gian thực từ dữ liệu keypoints được trích xuất bằng **MediaPipe**, sử dụng mô hình **TransformerEncoder** để phân loại các cử chỉ sau:

- 👌 OK
- ✊ Nắm đấm
- 🖐 Xòe tay
- ☝ 1 ngón
- ✌ 2 ngón
- 🤟 3 ngón
- 🖖 4 ngón
- 🖕 Cử chỉ bậy bạ

---

## 🎥 Video Demo
[![Watch the demo](demo_thumbnail.png)](demo_video.mp4)  
*Bấm vào hình để xem video demo kết quả chạy mô hình.*

---

## Data Sử Dụng
Nguồn: Dữ liệu keypoints được thu thập bằng MediaPipe Hands.
Định dạng: Mỗi mẫu là mảng (21, 3) gồm tọa độ (x, y, z) của 21 landmarks trên bàn tay.
Tiền xử lý:
Chuẩn hóa vị trí (dịch để cổ tay về gốc tọa độ).
Scale kích thước bàn tay về cùng tỷ lệ.
Số lớp: 8 lớp cử chỉ (OK, Nắm đấm, Xòe tay, 1, 2, 3, 4, Cử chỉ bậy bạ).

---

## Transformer Overview
Mô hình sử dụng TransformerEncoder để xử lý chuỗi keypoints (21 điểm trên bàn tay, mỗi điểm có tọa độ (x, y, z)).

Pipeline xử lý: Keypoints Sequence → Positional Encoding → TransformerEncoder → Pooling → Classifier
MediaPipe phát hiện bàn tay và trích xuất keypoints (21×3).
Chuẩn hóa dữ liệu keypoints (scale, translate).
Mỗi frame → 1 vector (sequence length = 21, feature dim = 3).
Positional Encoding thêm thông tin vị trí.
TransformerEncoder học mối quan hệ không gian giữa các điểm.
Pooling + Fully Connected Layer + Softmax → dự đoán nhãn.

---

## 📦 Cài đặt nhanh
Bạn có thể cài theo hai cách:

**1. Sử dụng `requirements.txt`**
```bash
pip install -r requirements.txt
```
**2. Sử dụng trực tiếp**
```bash
pip install torch mediapipe opencv-python numpy scikit-learn
```
***3. Deploy streamlit***
```bash
streamlit run app.py
```
