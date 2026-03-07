from ultralytics import YOLO
import cv2
import sys
import csv
import os

from collections import deque
VIDEO_PATH = sys.argv[1]
MODEL_PATH = sys.argv[2]
OUTPUT_DIR = sys.argv[3]
OUTPUT_VIDEO = os.path.join(OUTPUT_DIR,"abnormal_output.avi")
OUTPUT_CSV = os.path.join(OUTPUT_DIR,"collective_timeseries.csv")
FRAME_PREVIEW = os.path.join(OUTPUT_DIR,"current_frame.jpg")
CONFIDENCE_RATE = 0.25
ABNORMAL_CLASSES = [3,4,5,6]

# ===== 时序窗口大小 =====
TEMPORAL_WINDOW = 5 #时间窗口为5帧
ratio_buffer = deque(maxlen=TEMPORAL_WINDOW) #双端队列保存最近5帧数据

model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) #获取总帧数
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

#创建一个新的视频文件，用来保存检测结果
out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"XVID"),
    10,
    (width,height)
)

frame_idx = 0 #帧计数器
timeseries = [] #保存每一帧（平滑后）的异常比例数据

while cap.isOpened():
    ret,frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    results = model.predict(frame,conf=CONFIDENCE_RATE,verbose=False)
    behaviors = []
    if results[0].boxes is not None:

        boxes = results[0].boxes.xyxy.cpu().numpy() #取检测框目标，yolo结果返回tenser，转为numpy
        classes = results[0].boxes.cls.cpu().numpy() #取检测框的类别编号

        for box,cls in zip(boxes,classes): #将框坐标与行为类型打包成一个元组
            cls = int(cls)
            behaviors.append(cls) #以后用来统计异常比例
            x1,y1,x2,y2 = box.astype(int)
            color = (0,255,0) #正常行为画绿框
            if cls in ABNORMAL_CLASSES: #如果是异常种类，就画红框
                color = (0,0,255)
            cv2.rectangle(frame,(x1,y1),(x2,y2),color,2)

    total = len(behaviors)

    abnormal = sum(1 for b in behaviors if b in ABNORMAL_CLASSES)

    abnormal_ratio = abnormal/total if total>0 else 0

    # ===== 时序平滑 =====
    ratio_buffer.append(abnormal_ratio)
    ratio_smooth = sum(ratio_buffer)/len(ratio_buffer)

    timeseries.append([frame_idx,ratio_smooth])

    # ===== 根据异常比例改变颜色 =====
    text_color = (0,255,0)

    if ratio_smooth > 0.4:
        text_color = (0,0,255)
    elif ratio_smooth > 0.2:
        text_color = (0,255,255)

    cv2.putText(frame,
                f"Abnormal Ratio:{ratio_smooth:.2f}",
                (20,40), #文字位置
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                text_color,
                3)
    out.write(frame)

    # 实时帧保存（Qt实时显示）
    cv2.imwrite(FRAME_PREVIEW,frame) #把当前帧保存为一张图片文件
    progress = frame_idx/total_frames*100 #处理进度
    print(f"PROGRESS {progress:.2f} RATIO {ratio_smooth:.3f}",flush=True)

cap.release()
out.release()

with open(OUTPUT_CSV,"w",newline="") as f:

    writer = csv.writer(f)

    writer.writerow(["frame","ratio"])

    writer.writerows(timeseries)

print("DONE",flush=True)