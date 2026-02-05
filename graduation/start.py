import sys
import time

from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication
from ui import MyjinggaiApp

import siui
from siui.core import SiGlobal

#
# siui.gui.set_scale_factor(1)


def show_welcome_message(window):
    window.LayerRightMessageSidebar().send(
        title="欢迎使用井盖智能检测系统",
        text="在这里你可以对已有的井盖图片、视频进行快捷标注、管理",
        msg_type=1,
        icon=SiGlobal.siui.iconpack.get("ic_fluent_hand_wave_filled"),
        fold_after=5000
    )


if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MyjinggaiApp()
    window.show()

    timer = QTimer(window)
    timer.singleShot(500, lambda: show_welcome_message(window))

    sys.exit(app.exec_())
