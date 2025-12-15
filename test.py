import cv2
from ultralytics import YOLO
import torch
from pymongo import MongoClient
import numpy as np
from datetime import datetime
import face_recognition  # Dùng thư viện face_recognition để nhận diện và mã hóa khuôn mặt
import os
import uuid

# Kiểm tra thiết bị
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8l-face.pt')  # Bạn có thể thay đổi mô hình nếu cần
model.to(device)  # Đảm bảo mô hình chạy trên thiết bị phù hợp

# In thông tin thiết bị đang sử dụng
print(f"Model is running on: {device}")
if device == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Kết nối đến MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['dc']  # Tên Database
faces_collection = db['face_id']  # Tên Collection

# Khởi tạo camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Biến lưu tên và tuổi người dùng nhập
name = ""
age = ""

# Hàm vẽ ô nhập liệu
def draw_input_box(frame, text, pos):
    cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

# Hàm nhận dữ liệu người dùng nhập vào
def get_user_input(prompt, max_len=20):
    input_str = ""
    while True:
        # Hiển thị thông báo cho người dùng nhập dữ liệu
        cv2.putText(frame, f"{prompt}: {input_str}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('YOLOv8 - Face and People Detection', frame)

        # Chờ người dùng nhập và nhấn phím
        key = cv2.waitKey(1) & 0xFF
        if key == 13:  # Nhấn Enter để kết thúc nhập
            break
        elif key == 8:  # Nhấn Backspace để xóa ký tự
            input_str = input_str[:-1]
        elif 32 <= key <= 126:  # Chỉ cho phép nhập ký tự
            input_str += chr(key)

        # Giới hạn độ dài của chuỗi nhập
        if len(input_str) > max_len:
            input_str = input_str[:max_len]
        
    return input_str

# Vòng lặp chính để xử lý từng khung hình
while True:
    ret, frame = cap.read()

    if not ret:
        print("Không thể nhận frame từ camera")
        break

    # Nhận diện đối tượng từ ảnh
    results = model(frame)  # Mô hình YOLOv8 nhận diện đối tượng

    # Kiểm tra nếu có đối tượng được nhận diện
    if results:
        res = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(res, 'boxes', None)
        people_count = 0

        if boxes is not None:
            # Kiểm tra hộp bao quanh đối tượng và loại đối tượng
            try:
                xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else None
                cls = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else None

                if xyxy is not None and cls is not None:
                    for (x1, y1, x2, y2), c in zip(xyxy, cls):
                        if int(c) == 0:  # class 0 == người trong COCO
                            people_count += 1
                            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

                            # Cắt phần khuôn mặt từ frame
                            face = frame[y1:y2, x1:x2]

                            # Mã hóa khuôn mặt bằng thư viện face_recognition
                            encoding = face_recognition.face_encodings(face)

                            if encoding:  # Kiểm tra xem có khuôn mặt nào được mã hóa không
                                # Tạo một ID ngẫu nhiên cho ảnh
                                image_name = f"{str(uuid.uuid4())}.jpg"
                                image_path = os.path.join('face_images', image_name)

                                # Lưu ảnh khuôn mặt vào thư mục
                                if not os.path.exists('face_images'):
                                    os.makedirs('face_images')
                                cv2.imwrite(image_path, face)

                                # Hiển thị hộp nhập liệu cho tên và tuổi
                                name = get_user_input("Enter Name")
                                age = get_user_input("Enter Age")

                                # Nếu đã nhập đầy đủ thông tin, lưu vào MongoDB
                                if name and age:
                                    # Lưu thông tin vào MongoDB
                                    face_data = {
                                        "name": name,
                                        "age": age,
                                        "embedding": encoding[0].tolist(),  # Chuyển mảng numpy thành list trước khi lưu vào MongoDB
                                        "image_path": image_path,  # Lưu đường dẫn đến ảnh
                                        "created_at": datetime.utcnow()  # Lưu thời gian tạo
                                    }

                                    # Lưu dữ liệu vào MongoDB
                                    faces_collection.insert_one(face_data)
                                    print(f"Face saved to database with image path: {image_path}")
                            else:
                                print("No face encoding found, skipping this face.")

            except Exception as e:
                print(f"Error while processing boxes: {e}")

        # Hiển thị số lượng người được phát hiện
        cv2.putText(frame, f'People Detected: {people_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Hiển thị hình ảnh
    cv2.imshow('YOLOv8 - Face and People Detection', frame)

    # Dừng lại nếu nhấn phím 'q' hoặc capture lại khuôn mặt
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
