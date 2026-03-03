from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter, deque
import math
import os
import csv

# =========================
# 1. 路径与参数
# =========================
VIDEO_PATH = r"E:\DEEPlearning\4001067.mp4"
MODEL_PATH = r"E:\DEEPlearning\ultralytics-8.3.163\runs\detect\train7\weights\best.pt"

OUTPUT_VIDEO = "abnormal_output.avi"
OUTPUT_CSV = "collective_timeseries.csv"

CONF_THRES = 0.25

FPS = 0.5
WINDOW_SECONDS = 20
WINDOW_SIZE = int(FPS * WINDOW_SECONDS)

GRID_SIZE = 120

# =========================
# 2. 行为类别
# =========================
BEHAVIORS = {
    0: "look up",
    1: "reading",
    2: "writing",
    3: "using phone",
    4: "bow head",
    5: "sleeping",
    6: "turn head"
}

# 定义异常行为
ABNORMAL_CLASSES = [3,4,5,6]   # phone / bow head / sleep

# =========================
# 3. 工具函数
# =========================
def get_seat_id(cx, cy):
    gx = cx // GRID_SIZE
    gy = cy // GRID_SIZE
    return f"G{gx}_{gy}"

def behavior_ratio(seq, target):
    return seq.count(target) / len(seq) if seq else 0

def behavior_entropy(seq):
    if not seq:
        return 0
    cnt = Counter(seq)
    probs = [v / len(seq) for v in cnt.values()]
    return -sum(p * math.log2(p) for p in probs)

# =========================
# 4. 异常规则
# =========================
def detect_seat_anomaly(seq):
    anomalies = []

    if behavior_ratio(seq, 5) >= 0.6:
        anomalies.append("LONG_SLEEP")

    if behavior_ratio(seq, 3) >= 0.6:
        anomalies.append("LONG_PHONE")

    if behavior_ratio(seq, 4) >= 0.6:
        anomalies.append("LONG_HEAD_DOWN")

    switches = sum(seq[i] != seq[i - 1] for i in range(1, len(seq)))
    if switches >= 5:
        anomalies.append("FREQUENT_SWITCH")

    if behavior_entropy(seq) >= 1.5:
        anomalies.append("BEHAVIOR_CHAOS")

    return anomalies


def detect_collective_anomaly(abnormal_ratio):
    anomalies = []

    if abnormal_ratio >= 0.5:
        anomalies.append("SERIOUS_ABNORMAL")

    if abnormal_ratio >= 0.3:
        anomalies.append("WARNING_STATE")

    return anomalies


# =========================
# 5. 初始化
# =========================
model = YOLO(MODEL_PATH)
cap = cv2.VideoCapture(VIDEO_PATH)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

out = cv2.VideoWriter(
    OUTPUT_VIDEO,
    cv2.VideoWriter_fourcc(*"XVID"),
    1,
    (width, height)
)

seat_windows = {}
frame_idx = 0

# 保存时间序列
timeseries_data = []

# =========================
# 6. 主循环
# =========================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_idx += 1
    current_time = frame_idx / FPS
    frame_behaviors = []

    results = model.predict(
        source=frame,
        conf=CONF_THRES,
        verbose=False
    )

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        classes = results[0].boxes.cls.cpu().numpy()

        for box, cls_id in zip(boxes, classes):
            x1, y1, x2, y2 = box.astype(int)
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            cls_id = int(cls_id)

            frame_behaviors.append(cls_id)

            seat_id = get_seat_id(cx, cy)

            if seat_id not in seat_windows:
                seat_windows[seat_id] = deque(maxlen=WINDOW_SIZE)

            seat_windows[seat_id].append(cls_id)

            anomalies = []
            if len(seat_windows[seat_id]) >= WINDOW_SIZE:
                anomalies = detect_seat_anomaly(list(seat_windows[seat_id]))

            # 可视化单人异常
            if anomalies:
                color = (0, 0, 255)
                label = f"{seat_id} | {'/'.join(anomalies)}"
            else:
                color = (0, 255, 0)
                label = f"{seat_id} | {BEHAVIORS.get(cls_id, cls_id)}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # =========================
    # 群体统计
    # =========================
    total_count = len(frame_behaviors)
    abnormal_count = sum(1 for b in frame_behaviors if b in ABNORMAL_CLASSES)

    abnormal_ratio = abnormal_count / total_count if total_count > 0 else 0

    timeseries_data.append([
        frame_idx,
        round(current_time, 2),
        total_count,
        abnormal_count,
        round(abnormal_ratio, 3)
    ])

    collective = detect_collective_anomaly(abnormal_ratio)

    # =========================
    # 可视化群体信息
    # =========================
    cv2.putText(frame,
                f"Abnormal Ratio: {abnormal_ratio:.2f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 255, 0),
                3)

    if collective:
        cv2.putText(frame,
                    "COLLECTIVE: " + "/".join(collective),
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3)

    out.write(frame)
    cv2.imshow("Abnormal Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break


# =========================
# 7. 保存CSV
# =========================
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["frame", "time_sec", "total", "abnormal", "ratio"])
    writer.writerows(timeseries_data)

cap.release()
out.release()
cv2.destroyAllWindows()

print(f"✅ 视频已保存：{os.path.abspath(OUTPUT_VIDEO)}")
print(f"✅ 时间序列已保存：{os.path.abspath(OUTPUT_CSV)}")
