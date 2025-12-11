import cv2
from ultralytics import YOLO
import torch
import face_recognition  # Dùng thư viện face_recognition để nhận diện và so khớp khuôn mặt
import numpy as np
import os

# Kiểm tra thiết bị
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO('yolov8l-face.pt')  # Bạn có thể thay đổi mô hình nếu cần
model.to(device)  # Đảm bảo mô hình chạy trên thiết bị phù hợp

# In thông tin thiết bị đang sử dụng
print(f"Model is running on: {device}")
if device == 'cuda':
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Khởi tạo camera
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

# Dữ liệu khuôn mặt đã đăng ký
registered_faces = []  # Danh sách lưu trữ mảng đặc trưng khuôn mặt
names = []  # Danh sách tên người

# Đọc dữ liệu khuôn mặt đã đăng ký từ file nếu có
if os.path.exists('registered_faces.npy') and os.path.exists('names.npy'):
    registered_faces = list(np.load('registered_faces.npy', allow_pickle=True))
    names = list(np.load('names.npy', allow_pickle=True))

# Hàm lưu khuôn mặt đã đăng ký
def register_face(frame, name):
    rgb_frame = frame[:, :, ::-1]  # Chuyển ảnh thành RGB
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    if len(face_encodings) > 0:
        registered_faces.append(face_encodings[0])  # Lưu mảng đặc trưng của khuôn mặt
        names.append(name)  # Lưu tên người
        np.save('registered_faces.npy', registered_faces)  # Lưu vào file
        np.save('names.npy', names)  # Lưu tên vào file
        print(f"Đã đăng ký khuôn mặt của {name}")
    else:
        print("Không phát hiện khuôn mặt để đăng ký")

# Vòng lặp chính để xử lý từng khung hình
while True:
    ret, frame = cap.read()

    if not ret:
        print("Không thể nhận frame từ camera")
        break

    # Nhận diện khuôn mặt từ YOLO
    results = model(frame)  # Mô hình YOLOv8 nhận diện đối tượng

    # Kiểm tra nếu có đối tượng được nhận diện
    if results:
        res = results[0] if isinstance(results, (list, tuple)) else results
        boxes = getattr(res, 'boxes', None)
        face_locations = []
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

            except Exception as e:
                print(f"Error while processing boxes: {e}")

        # Xử lý nhận diện khuôn mặt
        rgb_frame = frame[:, :, ::-1]  # Chuyển ảnh thành RGB
        face_locations = face_recognition.face_locations(rgb_frame)  # Lấy vị trí khuôn mặt
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)  # Lấy đặc trưng khuôn mặt

        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(registered_faces, face_encoding)
            name = "Không nhận diện"

            if True in matches:
                first_match_index = matches.index(True)
                name = names[first_match_index]  # Lấy tên của người đã đăng ký
            else:
                name = "Khuôn mặt lạ"

            # Vẽ hộp bao quanh khuôn mặt và tên người
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Hiển thị số lượng người được phát hiện
        cv2.putText(frame, f'People Detected: {people_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Hiển thị hình ảnh
    cv2.imshow('YOLOv8 - Face and People Detection', frame)

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
