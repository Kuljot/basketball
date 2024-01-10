import os
import cv2
import sys
from ultralytics import YOLO


video_path = os.path.join('.',sys.argv[1])
model_path = os.path.join('.',sys.argv[2])
video_path_out = '{}_out.mp4'.format(video_path)

if not os.path.exists(video_path):
    raise Exception("Video path doesn't exist") 
if not os.path.exists(model_path):
    raise Exception("Model path doesn't exist")

cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
H, W, _ = frame.shape
VIDEOS_DIR = os.path.join('.')
out = cv2.VideoWriter(video_path_out, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))


# Load a model
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while ret:

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    out.write(frame)
    ret, frame = cap.read()

cap.release()
out.release()
cv2.destroyAllWindows()    

