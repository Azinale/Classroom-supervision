import cv2
from ultralytics import YOLO
import torch
import threading
import time
import numpy as np
import math

# --- CẤU HÌNH HỆ THỐNG ---
# Danh sách các camera: Có thể là số (USB) hoặc chuỗi RTSP (IP Camera)
# Ví dụ: SOURCES = [0, 1, "rtsp://admin:pass@192.168.1.5..."]
SOURCES = [0, 1]  

# Cấu hình hiển thị
IMG_SIZE = (640, 480) # Resize về kích thước này để hiển thị lưới cho đẹp
CONF_THRESHOLD = 0.5  # Độ tin cậy tối thiểu

# --- 1. CLASS XỬ LÝ CAMERA ĐA LUỒNG ---
class CameraStream:
    def __init__(self, source_id, index):
        self.id = index
        self.source = source_id
        # Thêm CAP_DSHOW nếu là USB cam trên Windows để fix lỗi
        if isinstance(source_id, int):
            self.cap = cv2.VideoCapture(source_id, cv2.CAP_DSHOW)
        else:
            self.cap = cv2.VideoCapture(source_id)
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def start(self):
        if self.started:
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.daemon = True # Tự động tắt khi chương trình chính tắt
        self.thread.start()
        return self

    def update(self):
        while self.started:
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame
            time.sleep(0.01) # Nghỉ nhẹ để giảm tải CPU

    def read(self):
        with self.read_lock:
            if not self.grabbed:
                return None
            return self.frame.copy()

    def stop(self):
        self.started = False
        if self.thread.is_alive():
            self.thread.join()
        self.cap.release()

# --- 2. KHỞI TẠO MODEL ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"System running on: {device}")

# Lưu ý: Với nhiều camera, nên dùng model nhẹ (yolov8n hoặc yolov8s) để đảm bảo FPS
# Nếu GPU mạnh (RTX 3060 trở lên) mới nên dùng yolov8l
print("Loading Model...")
model = YOLO('yolov8n.pt') # Khuyên dùng bản nano (n) hoặc small (s) cho real-time nhiều cam
model.to(device)

# --- 3. KHỞI TẠO CÁC LUỒNG CAMERA ---
streams = []
for i, src in enumerate(SOURCES):
    print(f"Khởi tạo Camera {i} từ nguồn: {src}")
    stream = CameraStream(src, i).start()
    streams.append(stream)

print("Hệ thống đã sẵn sàng. Nhấn 'q' để thoát.")

# --- 4. VÒNG LẶP CHÍNH ---
try:
    while True:
        frames = []
        valid_streams = []

        # B1: Lấy frame từ tất cả camera
        for stream in streams:
            frame = stream.read()
            if frame is not None:
                frames.append(frame)
                valid_streams.append(stream)
            else:
                # Tạo màn hình đen nếu mất tín hiệu
                blank = np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), np.uint8)
                cv2.putText(blank, f"Cam {stream.id} Lost", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                frames.append(blank)

        if not frames:
            continue

        # B2: Batch Inference (Gửi cả list frames vào model 1 lần)
        # Đây là chìa khóa để tăng tốc độ
        results = model(frames, verbose=False, classes=[0]) # class 0 là người

        # B3: Vẽ kết quả lên từng frame
        processed_frames = []
        for i, result in enumerate(results):
            # Lấy frame gốc để vẽ lên
            current_frame = frames[i]
            
            # Resize về kích thước chuẩn để ghép lưới (nếu cần)
            current_frame = cv2.resize(current_frame, IMG_SIZE)

            # Vẽ bounding box
            people_count = 0
            boxes = result.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                for box in xyxy:
                    people_count += 1
                    x1, y1, x2, y2 = map(int, box)
                    # Scale lại tọa độ nếu ảnh bị resize (Optional - code này giả định size khớp)
                    cv2.rectangle(current_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Vẽ thông tin lên góc
            cv2.putText(current_frame, f'CAM {i} | P: {people_count}', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            
            processed_frames.append(current_frame)

        # B4: Tạo lưới hiển thị (Grid Display)
        # Tính toán số hàng/cột
        n_cams = len(processed_frames)
        cols = math.ceil(math.sqrt(n_cams))
        rows = math.ceil(n_cams / cols)

        # Thêm ảnh đen vào nếu thiếu để đủ lưới chữ nhật
        while len(processed_frames) < rows * cols:
            processed_frames.append(np.zeros((IMG_SIZE[1], IMG_SIZE[0], 3), np.uint8))

        # Ghép ảnh
        row_images = []
        for r in range(rows):
            # Lấy các ảnh của hàng hiện tại
            row_imgs = processed_frames[r*cols : (r+1)*cols]
            # Ghép ngang (Horizontal)
            h_concat = cv2.hconcat(row_imgs)
            row_images.append(h_concat)
        
        # Ghép dọc (Vertical) các hàng lại
        final_grid = cv2.vconcat(row_images)

        # B5: Hiển thị
        cv2.imshow('He Thong Giam Sat Da Camera', final_grid)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Lỗi: {e}")

finally:
    # Dọn dẹp
    for s in streams:
        s.stop()
    cv2.destroyAllWindows()