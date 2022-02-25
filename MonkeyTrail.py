from PyQt5.QtWidgets import QApplication, QMainWindow
from pro.qt_3 import Ui_MainWindow
import argparse
import os
import glob
from pro.MonkeyTrail_Main  import MonkeyTrail_Step

class MainWindow(QMainWindow,Ui_MainWindow):

    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent=parent)
        ui = Ui_MainWindow()
        ui.setupUi(self)
        ui.radioButton_auto.setChecked(True)
        ui.radioButton_auto.toggle()
        ui.radioButton_man.setChecked(True)
        ui.radioButton_man.toggle()
        ui.spinBox_fps.setValue(10)
        ui.spinBox_back_man.setValue(40)
        ui.spinBox_interest.setValue(1)
        ui.start_button.setCheckable(True)
        ui.start_button.toggle()
        ui.start_button.clicked.connect(lambda :self.CalculateTax(ui))
        ui.start_button.setCheckable(False)

    def CalculateTax(self,ui):

        video_input = ui.video_input_box.toPlainText()
        config_path = ui.config_env_box.toPlainText()
        value_zone = ui.spinBox_interest.value()
        value_zone_uninterest = ui.spinBox_uninterest.value()
        value_fps = ui.spinBox_fps.value()
        if ui.radioButton_man.isChecked()==True:
            crop_flag = True
        if ui.radioButton_auto.isChecked()==True:
            crop_flag = False
        update_during = ui.spinBox_back_man.value()

        if not os.path.exists(video_input):
            print('The video path is incorrect!')
            sys.exit(app.exec_())
        if len(glob.glob(video_input + '/*.mp4'))==0:
            print("There's no video here : "+video_input)
            sys.exit(app.exec_())
        if not os.path.exists(config_path):
            os.makedirs(config_path)

        parser = argparse.ArgumentParser()
        parser.add_argument('--video_path', type=str, default=video_input)
        parser.add_argument('--output_path', type=str, default=config_path)
        parser.add_argument('--env', type=str, default='light1')
        parser.add_argument('--preferences', type=str, default='top1')
        parser.add_argument('--deep_learing_mode', type=str, default='yolo')
        parser.add_argument('--create_dataset', default=0)
        parser.add_argument('--save_frame_img_i', default=0)
        parser.add_argument('--background_during', default=update_during)
        parser.add_argument('--crop_video', default=crop_flag)
        parser.add_argument('--RoI', default=value_zone)
        parser.add_argument('--RoUI', default=value_zone_uninterest)
        parser.add_argument('--set_fps_num', default=value_fps)
        opt = parser.parse_args()
        g = MonkeyTrail_Step(**vars(opt))
        g.monkeytrail_process()


import sys
if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWindow()
    w.setWindowTitle("MONKEY")
    w.show()
    sys.exit(app.exec_())

