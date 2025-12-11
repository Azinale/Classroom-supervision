import face_recognition
import cv2
import numpy as np
from pymongo import MongoClient
import os

# Kết nối tới MongoDB
client = MongoClient('mongodb://localhost:27017/')
db = client['face_recognition_db']  # Tạo database
faces_collection = db['faces']  # Tạo collection lưu khuôn mặt

# Khởi tạo camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Hàm đăng ký khuôn mặt vào MongoDB
def register_face(frame, name):
    rgb_frame = frame[:, :, ::-1]  # Chuyển ảnh thành RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if len(face_encodings) > 0:
        encoding = face_encodings[0]  # Lấy mảng đặc trưng của khuôn mặt

        # Lưu khuôn mặt và tên vào MongoDB
        face_data = {
            "name": name,
            "encoding": encoding.tolist()  # Chuyển mảng numpy thành list trước khi lưu vào MongoDB
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

    # Hiển thị hình ảnh
    cv2.imshow('Đăng ký khuôn mặt', frame)

    # Đăng ký khuôn mặt khi nhấn phím 'r'
    if cv2.waitKey(1) & 0xFF == ord('r'):
        name = input("Nhập tên người cần đăng ký: ")
        register_face(frame, name)

    # Dừng lại nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
