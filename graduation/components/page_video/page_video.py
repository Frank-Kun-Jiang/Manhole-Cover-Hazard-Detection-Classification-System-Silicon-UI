import os
import shutil
from datetime import datetime

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import timm

from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import time
import warnings
import cv2
import numpy
from PyQt5.QtCore import QPointF, QRectF, Qt, QTimer, QUrl, pyqtSignal, QThread
from PyQt5.QtGui import QCursor, QIcon, QDesktopServices, QImage, QPixmap, QPainter, QPen, QFont
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QGraphicsBlurEffect, QLabel, QWidget, QVBoxLayout, QHBoxLayout, QRadioButton, QButtonGroup, \
    QMessageBox, QFileDialog, QSplitter, QComboBox, QSlider, QPushButton, QSizePolicy, QProgressBar

from graduation.components.database import ImageResultDatabase
from siui.components.combobox import SiComboBox

from siui.components.page import SiPage
from siui.components.progress_bar import SiProgressBar
from siui.components.widgets import (
    SiCheckBox,
    SiDenseHContainer,
    SiDraggableLabel,
    SiIconLabel,
    SiLabel,
    SiLongPressButton,
    SiPixLabel,
    SiPushButton,
    SiRadioButton,
    SiSimpleButton,
    SiSwitch,
    SiToggleButton,
)

from siui.core import  SiGlobal

# 自定义可点击的标签
class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        self.clicked.emit()
        super().mousePressEvent(event)

def images_to_video(image_folder, output_video_path, fps=30):
    # 获取图像文件列表
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]

    # 确定第一张图像的尺寸
    frame = cv2.imread(os.path.join(image_folder, images[0]))
    height, width, _ = frame.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # 遍历图像并写入视频
    for image in images:
        img_path = os.path.join(image_folder, image)
        frame = cv2.imread(img_path)
        video.write(frame)

    # 释放视频对象
    video.release()

def video_to_frames(video_path, output_folder):
    # 检查输出文件夹是否存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 打开视频文件
    cap = cv2.VideoCapture(video_path)
    frame_count = 0

    # 逐帧读取视频并保存为 PNG 文件
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_path = os.path.join(output_folder, f"frame_{frame_count:05d}.png")
        cv2.imwrite(frame_path, frame)

    # 释放视频对象
    cap.release()

def list_files_in_directory(directory):
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_list.append(os.path.join(root, file))
    return file_list

def clear_folder(folder_path):
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
        else:
            clear_folder(file_path)
            os.rmdir(file_path)

def delete_folder_if_exists(folder_path):
    # 检查文件夹是否存在
    if os.path.exists(folder_path):
        # 如果存在，先删除文件夹及其内容
        shutil.rmtree(folder_path)
        print(f"已删除文件夹：{folder_path}")
    else:
        print(f"文件夹不存在：{folder_path}")

# 新增：VideoProcessingWorker，将视频处理部分放到子线程中执行
class VideoProcessingWorker(QThread):
    progress_signal = pyqtSignal(float)
    finished_signal = pyqtSignal()

    def __init__(self, video_path, base_dir, interval, output_video_path, output_folder, final_folder, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.base_dir = base_dir
        self.interval = interval
        self.output_video_path = output_video_path
        self.output_folder = output_folder
        self.final_folder = final_folder

    def run(self):
        video_to_frames(self.video_path, self.output_folder)
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(base_dir, "components", "database", "results.db")
        # 实例化数据库管理对象
        db = ImageResultDatabase(db_path)

        yolo_weights_path = os.path.join(self.base_dir, "model_weight", "Yolo.pt")
        model = YOLO(yolo_weights_path)

        # 指定要遍历的文件夹路径
        process_path = self.output_folder
        profiles = list_files_in_directory(process_path)
        videoname=os.path.basename(self.video_path)

        # 要检查和删除的文件夹路径
        clear_folder(self.final_folder)

        # 忽略特定类型的警告
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        video_to_frames(self.video_path, self.final_folder)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        jjh_model = timm.create_model('vit_base_patch16_224', pretrained=False)
        local_weights_path = os.path.join(self.base_dir, "model_weight", "vit_base_patch16_224.pt")
        jjh_model = jjh_model.to(device)

        num_ftrs = jjh_model.head.in_features
        jjh_model.head = nn.Linear(num_ftrs, 5)
        jjh_model.load_state_dict(torch.load(local_weights_path))

        frame_count = 0
        model1_total_time = 0
        model2_total_time = 0
        total_profiles = len(profiles)
        if self.interval == 0:
            for profile in profiles:
                pro_image = Image.open(profile)
                frame_count += 1

                model1_start_time = time.time()
                pro_results = model(source=profile, show=False, save=False)
                model1_end_time = time.time()
                model1_execution_time = model1_end_time - model1_start_time
                model1_total_time += model1_execution_time

                if pro_results[0]:
                    for box in pro_results[0].boxes.xyxy:
                        pro_predictions = box.tolist()
                        ori_conf = pro_results[0].boxes.conf[0].item()
                        cropped_ori_image = pro_image.crop(
                            (pro_predictions[0], pro_predictions[1], pro_predictions[2], pro_predictions[3]))

                        model2_start_time = time.time()
                        jjh_model.eval()
                        with torch.no_grad():
                            images = test_transforms(cropped_ori_image)
                            images = torch.unsqueeze(images, 0).to(device)
                            outputs = jjh_model(images)
                            _, predicted = torch.max(outputs.data, 1)
                            predicted = predicted.numpy()
                            model2_end_time = time.time()
                            model2_execution_time = model2_end_time - model2_start_time
                            model2_total_time += model2_execution_time

                            if predicted.tolist() == [0]:
                                text = "broke"
                                font_color = (0, 0, 0)  # 黑色文字
                                bg_color = (255, 165, 0)  # 橙色边框
                            elif predicted.tolist() == [1]:
                                text = "circle"
                                font_color = (255, 165, 0)  # 橙色文字
                                bg_color = (255, 255, 255)  # 白色边框
                            elif predicted.tolist() == [2]:
                                text = "good"
                                font_color = (0, 255, 0)  # 绿色文字
                                bg_color = (255, 255, 255)  # 白色边框
                            elif predicted.tolist() == [3]:
                                text = "lose"
                                font_color = (255, 0, 0)  # 红色文字
                                bg_color = (255, 255, 255)  # 白色边框
                            elif predicted.tolist() == [4]:
                                text = "uncovered"
                                font_color = (255, 165, 0)  # 橙色文字
                                bg_color = (255, 255, 255)  # 白色边框

                            custom_position = (pro_predictions[0] + 2, pro_predictions[1] + 2)
                            frame_path = os.path.join(self.output_folder, f"frame_{frame_count:05d}.png")
                            final_path = os.path.join(self.final_folder, f"frame_{frame_count:05d}.png")
                            final_image = Image.open(final_path)
                            font_size = 20
                            draw = ImageDraw.Draw(final_image)
                            font = ImageFont.truetype("arial.ttf", font_size)
                            text_width, text_height = draw.textsize(text, font=font)
                            bg_x, bg_y = custom_position
                            bg_width = text_width + 10
                            bg_height = text_height + 5
                            draw.rectangle([bg_x, bg_y, bg_x + bg_width, bg_y + bg_height], fill=bg_color)
                            extra_rect_coords = (
                                pro_predictions[0], pro_predictions[1], pro_predictions[2], pro_predictions[3])
                            draw.rectangle(extra_rect_coords, outline=bg_color, width=3)
                            text_position = (bg_x + 5, bg_y + 2)
                            draw.text(text_position, text, font=font, fill=font_color)
                            draw.rectangle([bg_x, bg_y, bg_x + bg_width, bg_y + bg_height], fill=bg_color)
                            draw.text(text_position, text, font=font, fill=font_color)
                            final_image.save(final_path)

                final_image.save(frame_path)
                progress = (frame_count / total_profiles)
                self.progress_signal.emit(progress)
            fps = 24
            images_to_video(self.output_folder, self.output_video_path, fps)
            self.finished_signal.emit()
        else:
            for profile in profiles:
                frame_count += 1
                if frame_count % self.interval==1:
                    pro_image = Image.open(profile)
                    # ori_response = model.predict(ofile, confidence=40, overlap=0)
                    # # 解析 JSON 响应
                    # ori_predictions = ori_response.json()['predictions']
                    yolo_weights_path = os.path.join(base_dir, "model_weight", "Yolo.pt")
                    model = YOLO(yolo_weights_path)
                    pro_results = model(source=profile, show=False, save=False)

                    if pro_results[0]:
                        if pro_results[0]:
                            # print(pro_results[0].probs)
                            for box in pro_results[0].boxes.xyxy:
                                pro_predictions = box.tolist()
                                ori_conf = pro_results[0].boxes.conf[0].item()
                                print(ori_conf)
                                cropped_ori_image = pro_image.crop(
                                    (pro_predictions[0], pro_predictions[1], pro_predictions[2], pro_predictions[3]))

                                jjh_model.eval()
                                with torch.no_grad():
                                    # images = np.array(cropped_ori_image)
                                    images = test_transforms(cropped_ori_image)
                                    images = torch.unsqueeze(images, 0).to(device)
                                    # print(len(images))
                                    outputs = jjh_model(images)
                                    # print(outputs.data)
                                    _, predicted = torch.max(outputs.data, 1)
                                    # print("------------------------------------------")
                                    predicted = predicted.numpy()
                                    print("predicted---------", predicted)

                                    # print(f"MODEL执行时间：{model2_execution_time} 秒")

                                    if predicted == [0]:
                                        # 井盖破损
                                        text = "broke"
                                        font_color = (0, 0, 0)  # 黑色文字
                                        bg_color = (255, 165, 0)  # 橙色边框
                                    elif predicted == [1]:
                                        # 井圈破损
                                        text = "circle"
                                        font_color = (255, 165, 0)  # 橙色文字
                                        bg_color = (255, 255, 255)  # 白色边框
                                    elif predicted == [2]:
                                        # 井盖完好
                                        text = "good"
                                        font_color = (0, 255, 0)  # 绿色文字
                                        bg_color = (255, 255, 255)  # 白色边框
                                    elif predicted == [3]:
                                        # 井盖消失
                                        text = "lose"
                                        font_color = (255, 0, 0)  # 红色文字
                                        bg_color = (255, 255, 255)  # 白色边框
                                    elif predicted == [4]:
                                        # 井盖未盖
                                        text = "uncovered"
                                        font_color = (255, 165, 0)  # 橙色文字
                                        bg_color = (255, 255, 255)  # 白色边框

                                    # 自定义文本位置
                                    custom_position = (pro_predictions[0] + 2, pro_predictions[1] + 2)  # 自定义位置坐标

                                    output_folder = os.path.join("processed", text)

                                    draw_pro_image = Image.open(profile)

                                    draw = ImageDraw.Draw(draw_pro_image)

                                    # 加载字体
                                    font_size = 20
                                    font = ImageFont.truetype("arial.ttf", font_size)

                                    # 计算文本大小
                                    text_width, text_height = draw.textsize(text, font=font)

                                    # 计算背景矩形的位置和大小
                                    bg_x, bg_y = custom_position
                                    bg_width = text_width + 10  # 加上一些额外的空间
                                    bg_height = text_height + 5  # 加上一些额外的空间

                                    # 绘制背景矩形
                                    draw.rectangle([bg_x, bg_y, bg_x + bg_width, bg_y + bg_height], fill=bg_color)
                                    extra_rect_coords = (
                                        pro_predictions[0], pro_predictions[1], pro_predictions[2], pro_predictions[3])
                                    draw.rectangle(extra_rect_coords, outline=bg_color, width=3)

                                    # 在背景矩形上添加文本
                                    text_position = (bg_x + 5, bg_y + 2)  # 稍微偏移以使文本居中

                                    draw.text(text_position, text, font=font, fill=font_color)

                                    # 保存修改后的图像
                                    new_filename = "0_" + videoname +"_frame"+ str(frame_count)+"_"+ datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3] + ".png"
                                    final_path = os.path.join(output_folder, new_filename)
                                    draw_pro_image.save(final_path)
                                    db.insert_img_result(final_path, text)

                    else:
                        output_folder = os.path.join("processed", "undef")
                        new_filename = "0_" + videoname +"_frame"+ str(frame_count)+"_"+ datetime.now().strftime("%Y%m%d-%H%M%S-%f")[:-3] + ".png"
                        final_path = os.path.join(output_folder, new_filename)
                        pro_image.save(final_path)
                        db.insert_img_result(final_path, "undef")


                progress = (frame_count / total_profiles)
                self.progress_signal.emit(progress)
            self.finished_signal.emit()



class VideoRecognition(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.video_path = None  # 存储导入的视频文件路径
        # 模拟处理后视频文件路径（实际项目中请替换为处理后输出的视频文件路径）
        self.processed_video_path = None
        self.interval = 0

        # 主布局
        main_layout = QVBoxLayout(self)

        # 下部区域：导入视频按钮、开始处理按钮和下拉选择框
        bottom_layout = QHBoxLayout()
        self.btn_import = SiPushButton(self)
        self.btn_import.attachment().setText("导入视频")
        self.btn_import.clicked.connect(self.import_video)
        bottom_layout.addWidget(self.btn_import)

        self.btn_start = SiPushButton(self)
        self.btn_start.attachment().setText("开始处理")
        self.btn_start.clicked.connect(self.start_processing)
        bottom_layout.addWidget(self.btn_start)

        self.combo_interval = SiComboBox(self)
        self.combo_interval.addOption("处理整个视频，不另保存图片格式 (耗时最长)")
        self.combo_interval.addOption("2秒保存一帧")
        self.combo_interval.addOption("3秒保存一帧")
        self.combo_interval.addOption("5秒保存一帧")
        self.combo_interval.menu().setIndex(0)
        bottom_layout.addWidget(self.combo_interval)

        main_layout.addLayout(bottom_layout, 0.5)

        # 上部区域：左右两个视频预览框，左右排布，自适应大小
        preview_layout = QHBoxLayout()
        self.left_preview = ClickableLabel(self)
        self.right_preview = ClickableLabel(self)
        self.left_preview.setStyleSheet("""
                    background-color: #25222a;
                    border: 1px solid #4c4554;
                    border-radius: 10px;
                    color: white;
                """)
        self.right_preview.setStyleSheet("""
                    background-color: #25222a;
                    border: 1px solid #4c4554;
                    border-radius: 10px;
                    color: white;
                """)
        self.left_preview.setAlignment(Qt.AlignCenter)
        self.right_preview.setAlignment(Qt.AlignCenter)
        preview_layout.addWidget(self.left_preview)
        preview_layout.addWidget(self.right_preview)
        main_layout.addLayout(preview_layout,8)

        # 中间区域：进度条
        self.progress_bar = SiProgressBar(self)
        self.progress_bar.setValue(0)
        self.progress_bar.setTrackHeight(8)
        main_layout.addWidget(self.progress_bar)

        # 点击预览框调用系统默认播放器播放视频（这里只绑定左侧预览框）
        self.left_preview.clicked.connect(self.play_video)
        self.right_preview.clicked.connect(self.play_processed_video)

    def import_video(self):
        file, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "Video Files (*.mp4 *.avi *.mov)")
        if file:
            self.video_path = file
            # 导入视频后，仅在左侧预览框显示视频第一帧
            cap = cv2.VideoCapture(file)
            ret, frame = cap.read()
            cap.release()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                height, width, channels = frame.shape
                bytesPerLine = 3 * width
                qimg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                # 自适应左侧预览框大小
                scaled_pixmap = pixmap.scaled(self.left_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.left_preview.setPixmap(scaled_pixmap)
                self.progress_bar.setValue(0)
                # 重置右侧预览框（处理后预览为空）
                self.right_preview.clear()
            else:
                SiGlobal.siui.windows["MAIN_WINDOW"].LayerRightMessageSidebar().send(
                    "错误\n"
                    "无法加载视频第一帧",
                    msg_type=3,
                    fold_after=3000,
                )

    def play_video(self):
        if self.video_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.video_path))

    def play_processed_video(self):
        if self.video_path:
            QDesktopServices.openUrl(QUrl.fromLocalFile(self.processed_video_path))

    def start_processing(self):
        if not self.video_path:
            SiGlobal.siui.windows["MAIN_WINDOW"].LayerRightMessageSidebar().send(
                "提示\n"
                "请先导入视频！",
                msg_type=3,
                fold_after=3000,
            )
            return
        # 根据下拉框选择确定处理间隔
        SiGlobal.siui.windows["MAIN_WINDOW"].LayerRightMessageSidebar().send(
            "开始处理！\n"
            "请稍作等待",
            msg_type=1,
            fold_after=1000,
        )
        text = self.combo_interval.menu().current_value
        if "3秒" in text:
            interval = 72
        elif "2秒" in text:
            interval = 48
        elif "5秒" in text:
            interval = 120
        else:
            interval = 0

        # # 输入视频文件路径
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        # 输出视频路径
        output_video_path = os.path.join(base_dir, "processed","video", os.path.basename(self.video_path))

        self.processed_video_path = output_video_path

        #输出帧的文件夹路径
        output_folder = os.path.join(base_dir, "temp_frames")
        final_folder  = os.path.join(base_dir, "predicted_frames")

        clear_folder(output_folder)

        # 使用子线程处理视频，避免UI卡死
        self.worker = VideoProcessingWorker(
            video_path=self.video_path,
            base_dir=base_dir,
            interval=interval,
            output_video_path=output_video_path,
            output_folder=output_folder,
            final_folder=final_folder,
            parent=self
        )
        self.worker.progress_signal.connect(lambda progress: self.progress_bar.setValue(progress))
        self.worker.finished_signal.connect(self.on_processing_finished)
        self.worker.start()

    def on_processing_finished(self):
        self.progress_bar.setValue(100)
        self.update_processed_preview()
        SiGlobal.siui.windows["MAIN_WINDOW"].LayerRightMessageSidebar().send(
            "完成\n"
            "视频处理完毕",
            msg_type=1,
            fold_after=1000,
        )

    def update_processed_preview(self):
        if self.processed_video_path:
            cap = cv2.VideoCapture(self.processed_video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                # 为区分处理前后，我们在图像上叠加“Processed”文字
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # 将 frame 转为 QImage
                height, width, channels = frame.shape
                bytesPerLine = 3 * width
                qimg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg)
                # 使用 QPainter 在图像上叠加文字
                painter = QPainter(pixmap)
                painter.setRenderHint(QPainter.Antialiasing)
                pen = QPen(Qt.red)
                pen.setWidth(3)
                painter.setPen(pen)
                font = QFont("Arial",20, QFont.Bold)
                painter.setFont(font)
                painter.drawText(pixmap.rect(), Qt.AlignCenter, "Processed")
                painter.end()

                scaled_pixmap = pixmap.scaled(self.right_preview.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.right_preview.setPixmap(scaled_pixmap)
            else:
                SiGlobal.siui.windows["MAIN_WINDOW"].LayerRightMessageSidebar().send(
                    "提示\n"
                    "您选择了仅处理图像，请到管理界面查看检测识别结果",
                    msg_type=3,
                    fold_after=3000,
                )

    def reloadStyleSheet(self):
            super().reloadStyleSheet()
            # 标签
            # 文字标签
            self.demo_label.setStyleSheet("color: {}".format(SiGlobal.siui.colors["TEXT_A"]))
            self.demo_label_hinted.setStyleSheet("color: {}".format(SiGlobal.siui.colors["TEXT_A"]))
            self.demo_label_with_svg.setStyleSheet("color: {}".format(SiGlobal.siui.colors["TEXT_A"]))
            # 标签动画
            self.demo_label_ani.setColorTo(SiGlobal.siui.colors["INTERFACE_BG_E"])
            # 可拖动标签
            self.demo_draggable_label.setColorTo(SiGlobal.siui.colors["INTERFACE_BG_E"])