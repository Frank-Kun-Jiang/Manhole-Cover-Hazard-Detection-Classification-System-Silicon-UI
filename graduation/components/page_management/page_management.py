import os
from PyQt5.QtWidgets import QMenu
import pyperclip
from PyQt5.QtCore import Qt, QUrl
from PyQt5.QtGui import QDesktopServices, QPixmap, QIcon, QFontMetrics
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QListWidgetItem, QListWidget, QLabel, QPushButton

from graduation.components.page_management.image_context import ImageContextMenuDialog
from siui.templates.application.components.dialog.modal import SiModalDialog
from graduation.components.database import ImageResultDatabase
from siui.components import (
    SiDenseHContainer,
    SiDenseVContainer,
    SiFlowContainer,
    SiLabel,
    SiLineEdit,
    SiPushButton,
    SiScrollArea,
    SiSimpleButton,
)
from siui.components.combobox import SiComboBox
from siui.components.page import SiPage
from siui.core import SiColor
from siui.core import SiGlobal
from siui.core import Si


class management(SiPage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        main_layout = QVBoxLayout(self)

        # --------------------
        # 上部区域：下拉选择框 + 搜索区域
        # --------------------
        top_layout = SiDenseHContainer(self)
        top_layout.setAlignment(Qt.AlignCenter)
        top_layout.setFixedHeight(48)
        # top_layout.setSpacing(5)  # 减少各控件之间的水平间距

        # 下拉选择框（例如：全部、未处理、已处理）
        self.combo_category = SiComboBox(self)
        self.combo_category.addOption("全部类别")
        self.combo_category.addOption("井盖破损")
        self.combo_category.addOption("井圈破损")
        self.combo_category.addOption("全部完好")
        self.combo_category.addOption("井圈缺失")
        self.combo_category.addOption("井盖未盖")
        self.combo_category.menu().setIndex(0)
        self.combo_category.resize(300,32)
        top_layout.addWidget(self.combo_category)

        # 搜索框：按时间搜索
        self.time_search_label = SiLabel(self)
        # 在样式中增加左侧空隙
        self.time_search_label.setStyleSheet(f"color: {self.getColor(SiColor.TEXT_D)}; padding-left: 20px;")
        self.time_search_label.setText("按时间搜索:")
        self.time_search_label.adjustSize()
        top_layout.addWidget(self.time_search_label)

        self.search_time = SiLineEdit(self)
        self.search_time.resize(256, 32)
        self.search_time.reloadStyleSheet()
        self.search_time.colorGroup().assign(
            SiColor.INTERFACE_BG_B, self.getColor(SiColor.INTERFACE_BG_A))
        self.search_time.colorGroup().assign(
            SiColor.INTERFACE_BG_D, self.getColor(SiColor.INTERFACE_BG_C))
        top_layout.addWidget(self.search_time)

        # 搜索框：按视频文件搜索
        self.video_search_label = SiLabel(self)
        # 在样式中增加左侧空隙
        self.video_search_label.setStyleSheet(f"color: {self.getColor(SiColor.TEXT_D)}; padding-left: 20px;")
        self.video_search_label.setText("按视频文件搜索:")
        self.video_search_label.adjustSize()
        top_layout.addWidget(self.video_search_label)

        self.search_video = SiLineEdit(self)
        self.search_video.resize(256, 32)
        self.search_video.reloadStyleSheet()
        self.search_video.colorGroup().assign(
            SiColor.INTERFACE_BG_B, self.getColor(SiColor.INTERFACE_BG_A))
        self.search_video.colorGroup().assign(
            SiColor.INTERFACE_BG_D, self.getColor(SiColor.INTERFACE_BG_C))
        top_layout.addWidget(self.search_video)

        # 新增搜索按钮
        self.btn_search = SiPushButton(self)
        self.btn_search.attachment().setText("应用条件/刷新")
        self.btn_search.clicked.connect(self.on_search_clicked)
        top_layout.addWidget(self.btn_search)

        main_layout.addWidget(top_layout, 1)

        # --------------------
        # 下部区域：两个展示框（垂直排列，比例2:1）
        # --------------------
        bottom_layout = QVBoxLayout()

        # 未处理图像部分：上方添加说明标签，然后是展示框
        unprocessed_layout = QVBoxLayout()
        self.unprocessed_label = SiLabel(self)
        self.unprocessed_label.setStyleSheet("color: white;")
        self.unprocessed_label.setAlignment(Qt.AlignCenter)
        self.unprocessed_label.setText("未处理的图像")
        self.unprocessed_label.adjustSize()
        unprocessed_layout.addWidget(self.unprocessed_label)

        self.list_unprocessed = QListWidget()
        self.list_unprocessed.setViewMode(QListWidget.IconMode)
        icon_size = self.list_unprocessed.iconSize()
        if icon_size.isEmpty():
            icon_size = QPixmap(150, 150).size()
        self.list_unprocessed.setIconSize(icon_size)
        grid_size = QPixmap(160, 200).size()
        self.list_unprocessed.setGridSize(grid_size)
        self.list_unprocessed.setResizeMode(QListWidget.Adjust)
        self.list_unprocessed.setStyleSheet("""
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
        # self.list_unprocessed.itemActivated.connect(self.preview_image)
        self.list_unprocessed.itemClicked.connect(self.preview_image)
        # 同样，如果需要对未处理的列表设置右键菜单，也可类似设置
        self.list_unprocessed.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_unprocessed.customContextMenuRequested.connect(self.show_unprocessed_context_menu)
        unprocessed_layout.addWidget(self.list_unprocessed, 2)
        bottom_layout.addLayout(unprocessed_layout, 2)

        # 已处理图像部分：上方添加说明标签，然后是展示框
        processed_layout = QVBoxLayout()
        self.processed_label = SiLabel(self)
        self.processed_label.setStyleSheet("color: white;")
        self.processed_label.setAlignment(Qt.AlignCenter)
        self.processed_label.setText("已处理的图像")
        self.processed_label.adjustSize()
        processed_layout.addWidget(self.processed_label)

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

        self.list_processed.itemClicked.connect(self.preview_image)
        self.list_processed.setContextMenuPolicy(Qt.CustomContextMenu)
        self.list_processed.customContextMenuRequested.connect(self.show_processed_context_menu)


        processed_layout.addWidget(self.list_processed, 1)
        bottom_layout.addLayout(processed_layout, 1)

        main_layout.addLayout(bottom_layout, 18)
        self.ini_show()

    def preview_image(self, item: QListWidgetItem):
        file_path = item.data(Qt.ItemDataRole.UserRole)
        if file_path and os.path.exists(file_path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(file_path))

    def ini_show(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(base_dir, "components", "database", "results.db")
        db = ImageResultDatabase(db_path)
        results = db.query_results()  # 返回结果格式：(id, processed_image_path, predicted_label, timestamp)
        db.close()

        print(self.search_video.line_edit.text())
        unprocessed_paths = []
        processed_paths = []
        for row in results:
            processed_image_path = row[1]  # 第二列为处理后图片路径
            basename = os.path.basename(processed_image_path)
            if basename.startswith("0"):
                unprocessed_paths.append(processed_image_path)
            elif basename.startswith("1"):
                processed_paths.append(processed_image_path)

        # 调用 display_images 方法更新展示框
        self.display_images(unprocessed_paths, processed_paths)

    def on_search_clicked(self):
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        db_path = os.path.join(base_dir, "components", "database", "results.db")
        db = ImageResultDatabase(db_path)
        results = db.query_results()  # 结果格式：(id, processed_image_path, predicted_label, timestamp)
        db.close()

        # 获取下拉框选择的类别，并映射为 predicted_label
        selected_category = self.combo_category.menu().current_value
        category_filter = None
        if selected_category == "井盖破损":
            category_filter = "broke"
        elif selected_category == "井圈破损":
            category_filter = "circle"
        elif selected_category == "全部完好":
            category_filter = "good"
        elif selected_category == "井圈缺失":
            category_filter = "lose"
        elif selected_category == "井盖未盖":
            category_filter = "uncovered"
        # 如果选择"全部类别"，则 category_filter 仍为 None，不进行过滤

        # 获取搜索条件
        time_filter = self.search_time.line_edit.text().strip()
        video_filter = self.search_video.line_edit.text().strip()

        unprocessed_paths = []
        processed_paths = []
        for row in results:
            processed_image_path = row[1]
            predicted_label = row[2]  # 数据库中存储的预测标签
            basename = os.path.basename(processed_image_path)

            # 条件1：下拉选择条件过滤（如果未选择全部）
            if category_filter is not None and predicted_label != category_filter:
                continue

            # 条件2：如果时间搜索条件不为空，则文件名中必须包含该条件字符串
            if time_filter and time_filter not in basename:
                continue

            # 条件3：如果视频文件搜索条件不为空，则文件名中必须包含该条件字符串
            if video_filter and video_filter not in basename:
                continue

            # 根据文件名首字符区分未处理和已处理（假设：首字符 "0" 为未处理，"1" 为已处理）
            if basename.startswith("0"):
                unprocessed_paths.append(processed_image_path)
            elif basename.startswith("1"):
                processed_paths.append(processed_image_path)

        # 调用 display_images 方法更新展示框
        self.display_images(unprocessed_paths, processed_paths)
        print("搜索条件已应用，界面更新完成")

    def display_images(self, unprocessed_paths, processed_paths):
        """
        unprocessed_paths: list of file paths for 未处理的图像
        processed_paths: list of file paths for 已处理的图像
        """
        # 清空当前展示框
        self.list_unprocessed.clear()
        self.list_processed.clear()

        # 显示未处理图像到 self.list_unprocessed
        for file in unprocessed_paths:
            # 1) 加载原图
            pixmap = QPixmap(file)
            w = pixmap.width()
            h = pixmap.height()
            # 2) 居中裁切，得到正方形
            if w > h:
                x = (w - h) // 2
                pixmap = pixmap.copy(x, 0, h, h)
            elif h > w:
                y = (h - w) // 2
                pixmap = pixmap.copy(0, y, w, w)
            # 3) 统一缩放到展示框的 iconSize 大小
            scaled_pixmap = pixmap.scaled(
                self.list_unprocessed.iconSize(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            icon = QIcon(scaled_pixmap)
            # 4) 截断文件名，使用省略号
            fm = QFontMetrics(self.list_unprocessed.font())
            elided_text = fm.elidedText(os.path.basename(file), Qt.ElideRight, 150)
            item = QListWidgetItem(icon, elided_text)
            item.setTextAlignment(Qt.AlignCenter)
            # 将文件完整路径存储到项数据中，便于点击预览使用
            item.setData(Qt.ItemDataRole.UserRole, file)
            self.list_unprocessed.addItem(item)

        # 显示已处理图像到 self.list_processed
        for file in processed_paths:
            pixmap = QPixmap(file)
            w = pixmap.width()
            h = pixmap.height()
            if w > h:
                x = (w - h) // 2
                pixmap = pixmap.copy(x, 0, h, h)
            elif h > w:
                y = (h - w) // 2
                pixmap = pixmap.copy(0, y, w, w)
            scaled_pixmap = pixmap.scaled(
                self.list_processed.iconSize(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            icon = QIcon(scaled_pixmap)
            fm = QFontMetrics(self.list_processed.font())
            elided_text = fm.elidedText(os.path.basename(file), Qt.ElideRight, 150)
            item = QListWidgetItem(icon, elided_text)
            item.setTextAlignment(Qt.AlignCenter)
            item.setData(Qt.ItemDataRole.UserRole, file)
            self.list_processed.addItem(item)

    def show_processed_context_menu(self, pos):
        # 获取右键点击位置的项
        item = self.list_processed.itemAt(pos)
        if item:
            # 弹出模态对话框（参考你给出的 ModalDialogExample 的方式）
            dialog=ImageContextMenuDialog(self, item)
            dialog.refreshRequested.connect(self.on_search_clicked)  # 连接刷新信号
            SiGlobal.siui.windows["MAIN_WINDOW"].layerModalDialog().setDialog(dialog)

    def show_unprocessed_context_menu(self, pos):
        item = self.list_unprocessed.itemAt(pos)
        if item:
            dialog = ImageContextMenuDialog(self, item)
            dialog.refreshRequested.connect(self.on_search_clicked)
            SiGlobal.siui.windows["MAIN_WINDOW"].layerModalDialog().setDialog(dialog)





