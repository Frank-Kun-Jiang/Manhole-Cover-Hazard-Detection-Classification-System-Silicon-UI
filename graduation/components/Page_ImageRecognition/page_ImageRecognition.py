import pyperclip
from PyQt5.QtCore import Qt

from graduation.components.database import ImageResultDatabase
from siui.components import (
    SiDenseHContainer,
    SiDenseVContainer,
    SiFlowContainer,
    SiLabel,
    SiLineEdit,
    SiPushButton,
    SiScrollArea,
    SiSimpleButton, SiProgressBar,
)
import random
from siui.core import SiColor, GlobalFont
from siui.core import SiGlobal
from siui.gui import SiFont
from siui.components.combobox import SiComboBox
from siui.components.page import SiPage
from siui.core import SiColor
from siui.core import Si
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

from siui.components import SiTitledWidgetGroup, SiLabel, SiDenseHContainer, SiDenseVContainer, SiDividedHContainer, \
    SiDividedVContainer, SiFlowContainer, SiDraggableLabel, SiSimpleButton, SiPushButton, SiMasonryContainer
import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtGui import QDesktopServices, QIcon, QFontMetrics, QPixmap
from torchvision import datasets, transforms
import timm
import cv2
from torch.utils.data import ConcatDataset
from ultralytics import YOLO
import time
import warnings
from PIL import Image,ImageDraw, ImageFont

import os
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QLabel, QProgressBar, \
    QListWidgetItem, QFileDialog, QMessageBox
from PyQt5.QtCore import QSize, Qt, QTimer, QUrl


class imgRecognition(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # 用于存储导入的图片路径列表
        self.imported_images = []
        self.processed_images = []
        # 指定处理后图片存放的文件夹
        self.processed_folder = "processed"

        os.makedirs(self.processed_folder, exist_ok=True)

        # 主布局，分上下两个部分
        main_layout = QVBoxLayout(self)

        # 上部区域：左右两个大显示区
        display_layout = QHBoxLayout()

        # 左侧区域：导入的图片展示区
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        self.btn_import = SiPushButton(self)
        self.btn_import.attachment().setText("导入照片")
        self.btn_import.clicked.connect(self.import_images)
        left_layout.addWidget(self.btn_import)

        # 使用 QListWidget 以 IconMode 显示图片和文件名
        self.list_imported = QListWidget()
        self.list_imported.setViewMode(QListWidget.IconMode)  # PyQt5中直接使用 IconMode
        icon_size = QSize(150, 150)
        grid_size = QSize(160, 200)
        self.list_imported.setIconSize(icon_size)
        self.list_imported.setGridSize(grid_size)
        self.list_imported.setResizeMode(QListWidget.Adjust)  # PyQt5中直接使用 Adjust

        # 设置项的样式：圆角外边框、内边距及间距
        self.list_imported.setStyleSheet("""
                    QListWidget { 
                        border: 1px solid #4c4554; 
                        background-color: #25222a; 
                        border-radius: 10px;
                    }
                    QListWidget::item {
                        border: 1px solid #ccc;
                        margin: 5px;
                        padding: 5px;
                        border-radius: 8px;
                        background: #25222a;
                        color: white;
                    }
                    QScrollBar:vertical {
                        background: #4c4554;
                    }
                """)
        # 双击或激活项时预览图片
        self.list_imported.itemActivated.connect(self.preview_image)
        left_layout.addWidget(self.list_imported)
        display_layout.addWidget(left_widget)

        # 右侧区域：处理完成的图片展示区
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_label = SiLabel(self)
        right_label.setSiliconWidgetFlag(Si.AdjustSizeOnTextChanged)
        right_label.setText("处理完成的照片")
        right_label.setStyleSheet("color: white;")
        right_label.setAlignment(Qt.AlignCenter)
        right_layout.addSpacing(7)
        right_layout.addWidget(right_label)
        right_layout.addSpacing(8)

        self.list_processed = QListWidget()
        self.list_processed.setViewMode(QListWidget.IconMode)
        self.list_processed.setIconSize(icon_size)
        self.list_processed.setGridSize(grid_size)
        self.list_processed.setResizeMode(QListWidget.Adjust)
        self.list_processed.setStyleSheet("""
                    QListWidget { 
                        border: 1px solid #4c4554; 
                        background-color: #25222a; 
                        border-radius: 10px;
                    }
                    QListWidget::item {
                        border: 1px solid #ccc;
                        margin: 5px;
                        padding: 5px;
                        border-radius: 8px;
                        background: #25222a;
                        color: white;
                    }
                    QScrollBar:vertical {
                        background: #4c4554;
                    }
                """)
        self.list_processed.itemActivated.connect(self.preview_image)
        right_layout.addWidget(self.list_processed)
        display_layout.addWidget(right_widget)

        main_layout.addLayout(display_layout)

        # 下部区域：进度条和开始按钮
        bottom_layout = QHBoxLayout()
        self.progress_bar = SiProgressBar(self)
        self.progress_bar.setTrackHeight(8)
        bottom_layout.addWidget(self.progress_bar)
        self.btn_start = SiPushButton(self)
        self.btn_start.attachment().setText("开始处理")
        self.btn_start.clicked.connect(self.start_processing)
        bottom_layout.addWidget(self.btn_start)
        main_layout.addLayout(bottom_layout)


    def import_images(self):
        # 清空左右展示框的数据
        self.list_imported.clear()
        self.list_processed.clear()
        # 同时清空存储图片路径的列表
        self.imported_images = []
        self.processed_images = []

        files, _ = QFileDialog.getOpenFileNames(self, "选择图片", "", "Image Files (*.png *.jpg *.bmp)")
        if files:
            for file in files:
                self.imported_images.append(file)

                # 1) 加载原图到 QPixmap
                pixmap = QPixmap(file)
                w = pixmap.width()
                h = pixmap.height()

                # 2) 居中裁切，得到正方形
                if w > h:
                    # 宽大于高：截取中间部分
                    x = (w - h) // 2
                    pixmap = pixmap.copy(x, 0, h, h)
                elif h > w:
                    # 高大于宽：截取中间部分
                    y = (h - w) // 2
                    pixmap = pixmap.copy(0, y, w, w)
                # 如果 w == h 则不需要裁切

                # 3) 统一缩放到 iconSize 大小
                scaled_pixmap = pixmap.scaled(
                    self.list_imported.iconSize(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                icon = QIcon(scaled_pixmap)

                # 4) 对文件名进行截断处理
                fm = QFontMetrics(self.list_imported.font())
                elided_text = fm.elidedText(os.path.basename(file), Qt.TextElideMode.ElideRight, 150)

                item = QListWidgetItem(icon, elided_text)
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                # 将文件完整路径存储到用户角色数据中，便于预览时使用
                item.setData(Qt.ItemDataRole.UserRole, file)
                self.list_imported.addItem(item)

            # self.progress_bar.setMinimum(0)
            # self.progress_bar.setMaximum(len(self.imported_images))
            self.progress_bar.setValue(0)
            SiGlobal.siui.windows["MAIN_WINDOW"].LayerRightMessageSidebar().send(
                "完成\n"
                "照片导入完成",
                msg_type=1,
                fold_after= 1000,
            )

    def start_processing(self):
        if not self.imported_images:
            SiGlobal.siui.windows["MAIN_WINDOW"].LayerRightMessageSidebar().send(
                "提示\n"
                "请先导入照片！",
                msg_type=3,
                fold_after=3000,
            )
            return
        self.btn_start.setEnabled(False)
        self.current_index = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.process_next_image)
        self.timer.start(500)

    def process_next_image(self):
        # 加载预训练模型
        jjh_model = timm.create_model('vit_base_patch16_224', pretrained=False)
        # # 定义本地权重文件的路径
        base_dir =os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        local_weights_path = os.path.join(base_dir, "model_weight", "vit_base_patch16_224.pt")
        db_path = os.path.join(base_dir,"components", "database" , "results.db")
        # 实例化数据库管理对象
        db = ImageResultDatabase(db_path)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),  # 调整为模型期望的尺寸
            # transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        jjh_model = jjh_model.to(device)

        # 修改分类器
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, len(train_dataset.classes))
        # 修改分类器 vit和resnet不一样
        num_ftrs = jjh_model.head.in_features
        jjh_model.head = nn.Linear(num_ftrs, 5)
        jjh_model.load_state_dict(torch.load(local_weights_path))
        if self.current_index < len(self.imported_images):
            image_path = self.imported_images[self.current_index]

            pro_image = Image.open(image_path)
            # ori_response = model.predict(ofile, confidence=40, overlap=0)
            # # 解析 JSON 响应
            # ori_predictions = ori_response.json()['predictions']
            yolo_weights_path = os.path.join(base_dir, "model_weight", "Yolo.pt")
            model = YOLO(yolo_weights_path)
            pro_results = model(source=image_path, show=False, save=False)

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

                            draw_pro_image = Image.open(image_path)

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
                            new_filename = "0_" + time.strftime("%Y%m%d-%H%M%S") + ".png"
                            final_path = os.path.join(output_folder, new_filename)
                            draw_pro_image.save(final_path)
                            self.processed_images.append(final_path)
                            # 将结果写入数据库，只记录处理后图片路径和预测标签
                            db.insert_img_result(final_path, text)
            else:
                output_folder = os.path.join("processed", "undef")
                new_filename = "0_" + time.strftime("%Y%m%d-%H%M%S") + ".png"
                final_path = os.path.join(output_folder, new_filename)
                pro_image.save(final_path)
                self.processed_images.append(final_path)
                db.insert_img_result(final_path, "undef")


            self.current_index += 1
            self.progress_bar.setValue(self.current_index/len(self.imported_images))
        else:
            self.timer.stop()
            SiGlobal.siui.windows["MAIN_WINDOW"].LayerRightMessageSidebar().send(
                "完成\n"
                "处理完毕，请到管理页面进行筛选处理",
                msg_type=1,
                fold_after=3000,
            )
            self.load_processed_images()
            self.btn_start.setEnabled(True)

    def load_processed_images(self):
        self.list_processed.clear()
        for img_path in self.processed_images:
            # 1) 先加载原图
            pixmap = QPixmap(img_path)
            w = pixmap.width()
            h = pixmap.height()

            # 2) 居中裁切为正方形
            if w > h:
                x = (w - h) // 2
                pixmap = pixmap.copy(x, 0, h, h)
            elif h > w:
                y = (h - w) // 2
                pixmap = pixmap.copy(0, y, w, w)
            # 如果 w == h，则不需要裁切

            # 3) 缩放到与右侧列表相同的 iconSize
            scaled_pixmap = pixmap.scaled(
                self.list_processed.iconSize(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            icon = QIcon(scaled_pixmap)

            # 4) 截断文件名（使用省略号显示过长的名称）
            fm = QFontMetrics(self.list_processed.font())
            elided_text = fm.elidedText(os.path.basename(img_path),
                                        Qt.TextElideMode.ElideRight, 150)

            item = QListWidgetItem(icon, elided_text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            # 将完整路径存储到项数据中，便于预览使用
            item.setData(Qt.ItemDataRole.UserRole, img_path)
            self.list_processed.addItem(item)

    def preview_image(self, item: QListWidgetItem):
        # 获取存储的文件路径，并用系统默认应用打开
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path and os.path.exists(file_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))