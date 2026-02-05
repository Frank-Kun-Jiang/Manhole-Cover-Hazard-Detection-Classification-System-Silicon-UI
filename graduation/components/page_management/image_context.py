import os
import shutil
import sqlite3
import time
from PyQt5.QtCore import Qt, QUrl, pyqtSignal
from PyQt5.QtGui import QFont, QDesktopServices
from siui.templates.application.components.dialog.modal import SiModalDialog

from siui.components.widgets import SiLabel, SiPushButton
from siui.core import SiColor, SiGlobal

class ImageContextMenuDialog(SiModalDialog):
    refreshRequested = pyqtSignal()  # 添加刷新请求信号
    def __init__(self, parent, item):
        super().__init__(parent)
        self.setFixedWidth(500)
        self.item = item  # 存储当前操作的项
        self.item_path = item.data(Qt.ItemDataRole.UserRole)  # 保存文件路径，避免 item 被删除后再调用 data()
        # 假设数据库路径位于项目 components/database/results.db
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        self.db_path = os.path.join(base_dir, "components", "database", "results.db")
        self.base_dir = base_dir

        # 显示提示文本
        label1 = SiLabel(self)
        label1.setStyleSheet(f"color: {self.getColor(SiColor.TEXT_E)}")
        label1.setText("确定要对 " + self.item_path + " 进行修改吗")
        label1.adjustSize()
        self.contentContainer().addWidget(label1)

        # 修改类别按钮组：添加五个按钮
        buttons = [
            ("井盖破损", "broke"),
            ("井圈破损", "circle"),
            ("全部完好", "good"),
            ("井圈缺失", "lose"),
            ("井圈未盖", "uncovered")
        ]

        label2 = SiLabel(self)
        label2.setStyleSheet(f"color: {self.getColor(SiColor.TEXT_E)}")
        label2.setText("修改类别:")
        label2.adjustSize()
        self.buttonContainer().addWidget(label2)

        for text, target in buttons:
            btn = SiPushButton(self)
            btn.setFixedHeight(32)
            btn.attachment().setText(text)
            btn.colorGroup().assign(SiColor.BUTTON_PANEL, self.getColor(SiColor.INTERFACE_BG_D))
            # 单击后调用 do_action("modify", target)
            btn.clicked.connect(lambda _, t=target: self.do_action("modify", t))
            self.buttonContainer().addWidget(btn)

        label3 = SiLabel(self)
        label3.setStyleSheet(f"color: {self.getColor(SiColor.TEXT_E)}")
        label3.setText("文件操作:")
        label3.adjustSize()
        self.buttonContainer().addWidget(label3)

        # 按钮：新建副本
        button_copy = SiPushButton(self)
        button_copy.setFixedHeight(32)
        button_copy.attachment().setText("新建副本")
        button_copy.colorGroup().assign(SiColor.BUTTON_PANEL, self.getColor(SiColor.INTERFACE_BG_D))
        button_copy.clicked.connect(lambda: self.do_action("copy"))
        self.buttonContainer().addWidget(button_copy)

        # 按钮：删除项目
        button_delete = SiPushButton(self)
        button_delete.setFixedHeight(32)
        button_delete.attachment().setText("删除项目")
        button_delete.colorGroup().assign(SiColor.BUTTON_PANEL, self.getColor(SiColor.BUTTON_LONG_PRESS_PROGRESS))
        button_delete.clicked.connect(lambda: self.do_action("delete"))
        self.buttonContainer().addWidget(button_delete)

        # 按钮：标记已完成
        button_mark = SiPushButton(self)
        button_mark.setFixedHeight(32)
        button_mark.attachment().setText("标记已完成")
        button_mark.colorGroup().assign(SiColor.BUTTON_PANEL, self.getColor(SiColor.INTERFACE_BG_D))
        button_mark.clicked.connect(lambda: self.do_action("mark"))
        self.buttonContainer().addWidget(button_mark)

        # 按钮：取消
        button_cancel = SiPushButton(self)
        button_cancel.setFixedHeight(32)
        button_cancel.attachment().setText("取消")
        button_cancel.colorGroup().assign(SiColor.BUTTON_PANEL, self.getColor(SiColor.INTERFACE_BG_D))
        button_cancel.clicked.connect(lambda: self.do_nothing())
        self.buttonContainer().addWidget(button_cancel)

        # 刷新样式
        SiGlobal.siui.reloadStyleSheetRecursively(self)
        self.adjustSize()

    def do_action(self, action, target=None):
        if action == "modify":
            self.modify_item(self.item, target)
        elif action == "copy":
            self.copy_item(self.item)
        elif action == "delete":
            self.delete_item(self.item)
        elif action == "mark":
            self.mark_item_complete(self.item)
        # 关闭对话框
        # 发出刷新请求信号
        self.refreshRequested.emit()
        SiGlobal.siui.windows["MAIN_WINDOW"].layerModalDialog().closeLayer()

    def do_nothing(self):
        # 关闭对话框
        SiGlobal.siui.windows["MAIN_WINDOW"].layerModalDialog().closeLayer()

    def modify_item(self, item, target):
        # 修改类别：单击后将文件名的第一位改为"1"，移动到对应类别的文件夹，更新数据库记录，并刷新界面
        old_path = self.item_path  # 使用保存的文件路径
        basename = os.path.basename(old_path)
        if basename.startswith("0"):
            new_basename = "1" + basename[1:]
        else:
            new_basename = "1" + basename  # 强制修改为1开头
        new_folder = os.path.join(self.base_dir, "processed", target)
        if not os.path.exists(new_folder):
            os.makedirs(new_folder)
        new_path = os.path.join(new_folder, new_basename)
        try:
            shutil.move(old_path, new_path)
        except Exception as e:
            print("移动文件失败:", e)
            return
        # 更新数据库记录
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE image_results SET processed_image_path = ?, predicted_label = ? WHERE processed_image_path = ?", (new_path, target, old_path))
        conn.commit()
        conn.close()

    def copy_item(self, item):
        # 新建副本：复制当前项目对应的照片，新文件名为原文件名+_copied，存入同一文件夹，并插入数据库记录（predicted_label相同）
        old_path = self.item_path  # 使用保存的文件路径
        basename = os.path.basename(old_path)
        name, ext = os.path.splitext(basename)
        new_basename = name + "_copied" + ext
        new_path = os.path.join(os.path.dirname(old_path), new_basename)
        try:
            shutil.copy(old_path, new_path)
        except Exception as e:
            print("复制文件失败:", e)
            return
        # 获取原来的 predicted_label
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT predicted_label FROM image_results WHERE processed_image_path = ?", (old_path,))
        row = cursor.fetchone()
        predicted_label = row[0] if row else ""
        cursor.execute("INSERT INTO image_results (processed_image_path, predicted_label) VALUES (?, ?)", (new_path, predicted_label))
        conn.commit()
        conn.close()

    def delete_item(self, item):
        # 删除项目：删除当前项目对应的照片，删除数据库记录，刷新界面
        file_path = self.item_path  # 使用保存的文件路径
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print("删除文件失败:", e)
                return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM image_results WHERE processed_image_path = ?", (file_path,))
        conn.commit()
        conn.close()

    def mark_item_complete(self, item):
        # 标记已完成：修改文件名第一位为"1"，更新数据库记录，刷新界面
        old_path = self.item_path  # 使用保存的文件路径
        basename = os.path.basename(old_path)
        if basename.startswith("0"):
            new_basename = "1" + basename[1:]
        else:
            new_basename = basename
        new_path = os.path.join(os.path.dirname(old_path), new_basename)
        try:
            shutil.move(old_path, new_path)
        except Exception as e:
            print("移动文件失败:", e)
            return
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE image_results SET processed_image_path = ? WHERE processed_image_path = ?", (new_path, old_path))
        conn.commit()
        conn.close()

    def deleteLater(self):
        # 清理提示框（参考原有逻辑）
        SiGlobal.siui.windows["TOOL_TIP"].setNowInsideOf(None)
        SiGlobal.siui.windows["TOOL_TIP"].hide_()
        super().deleteLater()
