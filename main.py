import sys
import subprocess
import re
import os
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
class Worker(QThread):
    progress = pyqtSignal(float)
    ratio = pyqtSignal(float)
    finished = pyqtSignal()
    def __init__(self,video):
        super().__init__()
        self.video = video
    def run(self):
        process = subprocess.Popen(
            ["python","track_video.py",self.video],
            stdout=subprocess.PIPE,
            text=True
        )
        for line in process.stdout:
            if "PROGRESS" in line:
                p = float(re.findall(r'PROGRESS ([0-9.]+)',line)[0])
                r = float(re.findall(r'RATIO ([0-9.]+)',line)[0])
                self.progress.emit(p)
                self.ratio.emit(r)
        subprocess.run(["python","lstm_predict.py"])
        self.finished.emit()

class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("学生课堂异常行为检测系统")
        self.resize(1200,800)
        layout = QVBoxLayout()
        self.btn_select = QPushButton("选择视频")
        self.btn_run = QPushButton("开始检测")
        self.progress = QProgressBar()
        self.ratio_label = QLabel("全班异常比例：0.00")
        self.ratio_label.setFont(QFont("微软雅黑",16))
        self.video_label = QLabel()
        self.video_label.setFixedHeight(400)
        self.predict_label = QLabel()
        layout.addWidget(self.btn_select)
        layout.addWidget(self.btn_run)
        layout.addWidget(self.progress)
        layout.addWidget(self.ratio_label)
        layout.addWidget(self.video_label)
        layout.addWidget(self.predict_label)
        self.setLayout(layout)
        self.btn_select.clicked.connect(self.select_video)
        self.btn_run.clicked.connect(self.run_detection)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def select_video(self):
        file,_ = QFileDialog.getOpenFileName(
            self,
            "选择视频",
            "",
            "Video Files (*.mp4 *.avi)"
        )
        if file:
            self.video = file
            QMessageBox.information(
                self,
                "导入成功",
                f"文件路径：\n{file}\n导入成功"
            )
    def run_detection(self):
        self.worker = Worker(self.video)
        self.worker.progress.connect(self.update_progress)
        self.worker.ratio.connect(self.update_ratio)
        self.worker.finished.connect(self.show_result)
        self.worker.start()
        self.timer.start(100)

    def update_progress(self,p):
        self.progress.setValue(int(p))

    def update_ratio(self,r):
        self.ratio_label.setText(f"全班异常比例：{r:.2f}")
        if r>0.3:
            self.ratio_label.setStyleSheet("color:red")
        else:
            self.ratio_label.setStyleSheet("color:black")

    def update_frame(self):
        path = "output/current_frame.jpg"
        if os.path.exists(path):
            pix = QPixmap(path)
            self.video_label.setPixmap(
                pix.scaled(
                    self.video_label.width(),
                    self.video_label.height(),
                    Qt.KeepAspectRatio
                )
            )

    def show_result(self):
        self.timer.stop()
        img = QPixmap("output/predict.png")
        self.predict_label.setPixmap(img)

app = QApplication(sys.argv)
window = Window()
window.show()
sys.exit(app.exec_())