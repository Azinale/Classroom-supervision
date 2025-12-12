import cv2
import numpy as np
import face_recognition
from pymongo import MongoClient

# Kết nối tới MongoDB
client = MongoClient('mongodb://localhost:27017/')  # Sử dụng MongoDB local
db = client['dc']  # Tên Database
faces_collection = db['face_id']  # Tên Collection chứa khuôn mặt đã đăng ký

# Hàm so sánh khuôn mặt từ hình ảnh với các khuôn mặt trong cơ sở dữ liệu
def recognize_face(frame):
    rgb_frame = frame[:, :, ::-1]  # Chuyển ảnh thành RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = []
        
        # Lấy tất cả các khuôn mặt trong MongoDB
        registered_faces = faces_collection.find()

        for face in registered_faces:
            # So sánh khuôn mặt trong ảnh với những khuôn mặt đã đăng ký
            stored_encoding = np.array(face['embedding'])
            match = face_recognition.compare_faces([stored_encoding], face_encoding)
            if match[0]:
                name = face['name']
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                return frame, name

        # Nếu không nhận diện được khuôn mặt
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.putText(frame, "Người lạ", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    
    return frame, "Không nhận diện"

# Khởi tạo camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Vòng lặp chính để xử lý từng khung hình
while True:
    ret, frame = cap.read()

    if not ret:
        print("Không thể nhận frame từ camera")
        break

    # Điểm danh và nhận diện khuôn mặt
    frame, name = recognize_face(frame)

    # Hiển thị kết quả
    cv2.imshow('Điểm danh khuôn mặt', frame)

    # Dừng lại nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
