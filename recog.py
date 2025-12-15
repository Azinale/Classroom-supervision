import cv2
from ultralytics import YOLO
import torch

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

            except Exception as e:
                print(f"Error while processing boxes: {e}")

        # Hiển thị số lượng người được phát hiện
        cv2.putText(frame, f'People Detected: {people_count}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Hiển thị hình ảnh
    cv2.imshow('YOLOv8 - Face and People Detection', frame)

    # Dừng lại nếu nhấn phím 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
