# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'D:\test\MonkeyTrail-GUI\pro\qt_3.ui'
#
# Created by: PyQt5 UI code generator 5.15.6
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(326, 585)
        font = QtGui.QFont()
        font.setItalic(False)
        font.setUnderline(False)
        font.setStrikeOut(False)
        MainWindow.setFont(font)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(50, 20, 211, 51))
        font = QtGui.QFont()
        font.setPointSize(20)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.label_5.setFont(font)
        self.label_5.setObjectName("label_5")
        self.spinBox_fps = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_fps.setGeometry(QtCore.QRect(50, 292, 46, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.spinBox_fps.setFont(font)
        self.spinBox_fps.setObjectName("spinBox_fps")
        self.label_42 = QtWidgets.QLabel(self.centralwidget)
        self.label_42.setGeometry(QtCore.QRect(200, 322, 51, 41))
        font = QtGui.QFont()
        font.setFamily("Agency FB")
        font.setPointSize(10)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.label_42.setFont(font)
        self.label_42.setObjectName("label_42")
        self.label_6 = QtWidgets.QLabel(self.centralwidget)
        self.label_6.setGeometry(QtCore.QRect(20, 232, 91, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.label_6.setFont(font)
        self.label_6.setObjectName("label_6")
        self.label_41 = QtWidgets.QLabel(self.centralwidget)
        self.label_41.setGeometry(QtCore.QRect(20, 322, 121, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.label_41.setFont(font)
        self.label_41.setObjectName("label_41")
        self.label_40 = QtWidgets.QLabel(self.centralwidget)
        self.label_40.setGeometry(QtCore.QRect(20, 402, 171, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.label_40.setFont(font)
        self.label_40.setObjectName("label_40")
        self.spinBox_interest = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_interest.setGeometry(QtCore.QRect(200, 372, 46, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.spinBox_interest.setFont(font)
        self.spinBox_interest.setObjectName("spinBox_interest")
        self.video_input_box = QtWidgets.QTextEdit(self.centralwidget)
        self.video_input_box.setGeometry(QtCore.QRect(150, 92, 141, 41))
        self.video_input_box.setObjectName("video_input_box")
        self.groupBox_2 = QtWidgets.QGroupBox(self.centralwidget)
        self.groupBox_2.setGeometry(QtCore.QRect(120, 242, 121, 41))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setItalic(False)
        font.setUnderline(False)
        font.setStrikeOut(False)
        self.groupBox_2.setFont(font)
        self.groupBox_2.setTitle("")
        self.groupBox_2.setObjectName("groupBox_2")
        self.radioButton_man = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_man.setGeometry(QtCore.QRect(10, 10, 61, 19))
        self.radioButton_man.setObjectName("radioButton_man")
        self.radioButton_auto = QtWidgets.QRadioButton(self.groupBox_2)
        self.radioButton_auto.setGeometry(QtCore.QRect(70, 10, 41, 19))
        self.radioButton_auto.setObjectName("radioButton_auto")
        self.config_env_box = QtWidgets.QTextEdit(self.centralwidget)
        self.config_env_box.setGeometry(QtCore.QRect(150, 156, 141, 41))
        self.config_env_box.setObjectName("config_env_box")
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setGeometry(QtCore.QRect(10, 202, 271, 16))
        self.line.setFrameShape(QtWidgets.QFrame.HLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.label_11 = QtWidgets.QLabel(self.centralwidget)
        self.label_11.setGeometry(QtCore.QRect(20, 292, 31, 21))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.label_11.setFont(font)
        self.label_11.setObjectName("label_11")
        self.start_button = QtWidgets.QPushButton(self.centralwidget)
        self.start_button.setGeometry(QtCore.QRect(20, 460, 131, 51))
        font = QtGui.QFont()
        font.setPointSize(10)
        font.setBold(True)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(75)
        font.setStrikeOut(False)
        self.start_button.setFont(font)
        self.start_button.setObjectName("start_button")
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(20, 362, 151, 41))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(9)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.label_8.setFont(font)
        self.label_8.setObjectName("label_8")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(20, 86, 121, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.spinBox_back_man = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_back_man.setGeometry(QtCore.QRect(150, 332, 46, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.spinBox_back_man.setFont(font)
        self.spinBox_back_man.setObjectName("spinBox_back_man")
        self.spinBox_uninterest = QtWidgets.QSpinBox(self.centralwidget)
        self.spinBox_uninterest.setGeometry(QtCore.QRect(200, 412, 46, 22))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.spinBox_uninterest.setFont(font)
        self.spinBox_uninterest.setObjectName("spinBox_uninterest")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(20, 150, 121, 51))
        font = QtGui.QFont()
        font.setFamily("黑体")
        font.setPointSize(8)
        font.setBold(False)
        font.setItalic(False)
        font.setUnderline(False)
        font.setWeight(50)
        font.setStrikeOut(False)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 326, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_5.setText(_translate("MainWindow", "MonketTrail"))
        self.label_42.setText(_translate("MainWindow", "min/time"))
        self.label_6.setText(_translate("MainWindow", "Crop_video"))
        self.label_41.setText(_translate("MainWindow", "Update Frequency"))
        self.label_40.setText(_translate("MainWindow", "Region of Uinterest"))
        self.radioButton_man.setText(_translate("MainWindow", "Yes"))
        self.radioButton_auto.setText(_translate("MainWindow", "No"))
        self.label_11.setText(_translate("MainWindow", "Fps"))
        self.start_button.setText(_translate("MainWindow", "START"))
        self.label_8.setText(_translate("MainWindow", "Region of Interest"))
        self.label.setText(_translate("MainWindow", "Video Input Path"))
        self.label_2.setText(_translate("MainWindow", "Data Output Path"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())