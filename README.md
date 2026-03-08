# 在项目目录下创建虚拟环境
python -m venv venv
# 激活虚拟环境
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate

在虚拟环境中运行：
pip install --upgrade pip
pip install -r requirements.txt

使用 GUI 界面：python ui/main.py

1.基于yolo11n.pt训练的模型，位于model/，dataset来源于https://github.com/Whiffe/SCB-dataset

2.Student Classroom Anomaly Detection System 是一个基于深度学习的视频分析系统，用于自动识别课堂中学生的行为并检测异常情况。

3.系统整体流程如下：导入课堂视频 -> YOLO行为识别 -> 行为统计分析 -> 群体异常检测 -> 时间序列数据生成 -> LSTM趋势预测 -> 桌面客户端可视化

4.系统基于目标检测模型识别学生行为，包括：
  look up（抬头听讲）
  reading（看书）
  writing（写字）
  using phone（玩手机）
  bow head（低头）
  sleeping（睡觉）
  turn head（转头）
	
5.系统通过时间窗口分析学生行为序列，实现以下异常检测：
  长时间睡觉
  长时间玩手机
  长时间低头
  频繁行为切换
  行为混乱
	
6.系统实时统计全班异常行为比例：
  异常比例 >0.3：预警状态
  异常比例 >0.5：严重异常
  客户端会通过颜色变化进行提醒
	
7.系统使用LSTM模型对异常比例时间序列进行建模，实现：
  课堂状态趋势预测
  行为变化趋势分析
