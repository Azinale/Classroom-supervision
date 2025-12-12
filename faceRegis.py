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

    # Phát hiện các khuôn mặt trong ảnh
    face_locations = face_recognition.face_locations(rgb_frame)
    
    # Chỉ tiến hành nếu phát hiện được khuôn mặt
    if len(face_locations) > 0:
        # Lấy đặc trưng khuôn mặt
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Lưu ảnh khuôn mặt
        image_name = f"faces/{name.lower().replace(' ', '_')}.jpg"
        cv2.imwrite(image_name, frame)

        # Lưu vào MongoDB
        face_data = {
            "name": name,
            "age": age,
            "embedding": face_encodings[0].tolist(),  # Chuyển mảng numpy thành list trước khi lưu vào MongoDB
            "image_path": image_name,  # Lưu đường dẫn đến ảnh
            "created_at": datetime.utcnow()  # Lưu thời gian tạo
        }

        faces_collection.insert_one(face_data)
        print(f"Đã đăng ký khuôn mặt của {name}")
    else:
        print("Không phát hiện khuôn mặt để đăng ký")

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
