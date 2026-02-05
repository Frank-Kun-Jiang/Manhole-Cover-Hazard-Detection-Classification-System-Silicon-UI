import icons
from components.page_about import About
from components.page_homepage import ExampleHomepage
from components.page_icons import ExampleIcons
from components.Page_ImageRecognition import imgRecognition
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QDesktopWidget

import siui
from graduation.components.page_management import management
from graduation.components.page_video import VideoRecognition
from siui.core import SiColor, SiGlobal
from siui.templates.application.application import SiliconApplication

# 载入图标
siui.core.globals.SiGlobal.siui.loadIcons(
    icons.IconDictionary(color=SiGlobal.siui.colors.fromToken(SiColor.SVG_NORMAL)).icons
)


class MyjinggaiApp(SiliconApplication):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        screen_geo = QDesktopWidget().screenGeometry()
        self.setMinimumSize(1024, 380)
        self.resize(1366, 916)
        self.move((screen_geo.width() - self.width()) // 2, (screen_geo.height() - self.height()) // 2)
        self.layerMain().setTitle("井盖智能检测系统")
        self.setWindowTitle("井盖智能检测系统")
        self.setWindowIcon(QIcon("./img/avatar2.png"))

        self.layerMain().addPage(ExampleHomepage(self),
                                 icon=SiGlobal.siui.iconpack.get("ic_fluent_home_filled"),
                                 hint="主页", side="top")
        self.layerMain().addPage(imgRecognition(self),
                                 icon=SiGlobal.siui.iconpack.get("ic_fluent_table_image_filled"),
                                 hint="图片批量识别", side="top")
        self.layerMain().addPage(VideoRecognition(self),
                                 icon=SiGlobal.siui.iconpack.get("ic_fluent_video_filled"),
                                 hint="视频识别", side="top")
        self.layerMain().addPage(management(self),
                                 icon=SiGlobal.siui.iconpack.get("ic_fluent_poll_regular"),
                                 hint="综合管理", side="top")

        self.layerMain().addPage(About(self),
                                 icon=SiGlobal.siui.iconpack.get("ic_fluent_info_filled"),
                                 hint="关于", side="bottom")

        self.layerMain().setPage(0)

        SiGlobal.siui.reloadAllWindowsStyleSheet()
