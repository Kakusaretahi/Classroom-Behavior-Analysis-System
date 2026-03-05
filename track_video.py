from ultralytics import YOLO
import cv2
import sys
import csv
import os

VIDEO_PATH = sys.argv[1]

MODEL_PATH = "model/best.pt"

OUTPUT_VIDEO = "output/abnormal_output.avi"
OUTPUT_CSV = "output/collective_timeseries.csv"

FRAME_PREVIEW = "output/current_frame.jpg"

CONF_THRES = 0.25

ABNORMAL_CLASSES = [3,4,5,6]

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"XVID"),
    10,
    (width,height)
)

frame_idx = 0
timeseries = []

while cap.isOpened():

    ret,frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    results = model.predict(frame,conf=CONF_THRES,verbose=False)

    behaviors = []

    if results[0].boxes is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box,cls in zip(boxes,classes):

            cls = int(cls)
            behaviors.append(cls)

            x1,y1,x2,y2 = box.astype(int)

            color = (0,255,0)

            if cls in ABNORMAL_CLASSES:
                color = (0,0,255)

            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

    total = len(behaviors)

    abnormal = sum(1 for b in behaviors if b in ABNORMAL_CLASSES)

    ratio = abnormal/total if total>0 else 0

    timeseries.append([frame_idx,ratio])

    cv2.putText(frame,f"Abnormal Ratio:{ratio:.2f}",(20,40),
                cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),3)

    out.write(frame)

    # 实时帧保存
    cv2.imwrite(FRAME_PREVIEW,frame)

    progress = frame_idx/total_frames*100

    print(f"PROGRESS {progress:.2f} RATIO {ratio:.3f}",flush=True)

cap.release()
out.release()

with open(OUTPUT_CSV,"w",newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["frame","ratio"])

    writer.writerows(timeseries)

print("DONE",flush=True)