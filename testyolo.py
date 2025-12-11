# load libraries
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
import cv2  # OpenCV for camera access
from PIL import Image
import numpy as np

# download model
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")

# load model
model = YOLO(model_path)

# initialize camera
cap = cv2.VideoCapture(0)  # 0 for the default camera

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # read a frame from the camera
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # convert the frame to PIL Image for YOLO model
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # inference
    output = model(pil_image)
    results = Detections.from_ultralytics(output[0])

    # optional: draw bounding boxes or display results on the frame
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    cv2.imshow('Face Detection', frame)

    # break the loop when the user presses 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# release the camera and close windows
cap.release()
cv2.destroyAllWindows()
