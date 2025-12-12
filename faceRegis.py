import cv2
import numpy as np
import face_recognition
from pymongo import MongoClient
import os
from datetime import datetime
from ultralytics import YOLO
import uuid

# Kết nối đến MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['dc']  # Tên Database
faces_collection = db['face_id']  # Tên Collection

# Khởi tạo camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Khởi tạo mô hình YOLOv8
model = YOLO('yolov8l-face.pt')  # Dùng mô hình YOLOv8 để nhận diện khuôn mặt

# Tạo thư mục lưu ảnh nếu chưa tồn tại
os.makedirs("faces", exist_ok=True)

# Hàm đăng ký khuôn mặt vào MongoDB
def register_face(frame, name, age):
    rgb_frame = frame[:, :, ::-1]  # Chuyển ảnh thành RGB

    # Phát hiện khuôn mặt từ YOLO
    results = model(frame)  # Mô hình YOLOv8 nhận diện đối tượng
    face_locations = []

    # Kiểm tra nếu YOLO phát hiện khuôn mặt
    if results:
        res = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(res, 'boxes', None)

        if boxes is not None:
            xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes, 'xyxy') else None
            cls = boxes.cls.cpu().numpy() if hasattr(boxes, 'cls') else None

            if xyxy is not None and cls is not None:
                for (x1, y1, x2, y2), c in zip(xyxy, cls):
                    if int(c) == 0:  # class 0 == người trong COCO
                        face_locations.append((y1, x2, y2, x1))  # Chuyển đổi từ (top, right, bottom, left)

    # Nếu phát hiện khuôn mặt từ YOLO, tiến hành tạo encoding khuôn mặt
    if len(face_locations) > 0:
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        if len(face_encodings) > 0:
            encoding = face_encodings[0]  # Lấy mảng đặc trưng của khuôn mặt

            # Lưu ảnh khuôn mặt
            image_name = f"faces/{name.lower().replace(' ', '_')}.jpg"
            cv2.imwrite(image_name, frame)

            # Lưu vào MongoDB
            face_data = {
                "name": name,
                "age": age,
                "embedding": encoding.tolist(),  # Chuyển mảng numpy thành list trước khi lưu vào MongoDB
                "image_path": image_name,  # Lưu đường dẫn đến ảnh
                "created_at": datetime.utcnow()  # Lưu thời gian tạo
            }

            faces_collection.insert_one(face_data)
            print(f"Đã đăng ký khuôn mặt của {name}")
        else:
            print("Không phát hiện khuôn mặt để đăng ký")
    else:
        print("YOLO không phát hiện khuôn mặt")

# Vòng lặp chính để xử lý từng khung hình
while True:
    ret, frame = cap.read()

    if not ret:
        print("Không thể nhận frame từ camera")
        break

    # Hiển thị ô nhập tên và tuổi
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Nhập tên và tuổi:', (10, 30), font, 1, (0, 255, 255), 2)

    # Hiển thị panel nhập tên
    cv2.imshow('Đăng ký khuôn mặt', frame)

    # Nhấn phím 'r' để đăng ký khuôn mặt
    if cv2.waitKey(1) & 0xFF == ord('r'):
        name = input("Nhập tên người cần đăng ký: ")
        age = int(input("Nhập tuổi: "))
        register_face(frame, name, age)

    # Dừng lại nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
