# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '1.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1048, 831)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.setGeometry(QtCore.QRect(10, 540, 451, 231))
        self.tabWidget.setToolTip("")
        self.tabWidget.setObjectName("tabWidget")
        self.tab = QtWidgets.QWidget()
        self.tab.setObjectName("tab")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.tab)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 421, 201))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(10)
        self.verticalLayout.setObjectName("verticalLayout")
        self.groupBox_2 = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        self.groupBox_2.setObjectName("groupBox_2")
        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.groupBox_2)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(9, 10, 401, 51))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.horizontalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.Rota_label = QtWidgets.QLabel(self.horizontalLayoutWidget_3)
        self.Rota_label.setObjectName("Rota_label")
        self.horizontalLayout_3.addWidget(self.Rota_label)
        self.Rotasld = QtWidgets.QSlider(self.horizontalLayoutWidget_3)
        self.Rotasld.setMinimum(0)
        self.Rotasld.setMaximum(360)
        self.Rotasld.setSingleStep(5)
        self.Rotasld.setTracking(True)
        self.Rotasld.setOrientation(QtCore.Qt.Horizontal)
        self.Rotasld.setInvertedAppearance(False)
        self.Rotasld.setInvertedControls(False)
        self.Rotasld.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.Rotasld.setObjectName("Rotasld")
        self.horizontalLayout_3.addWidget(self.Rotasld)
        self.horizontalLayout_3.setStretch(0, 2)
        self.horizontalLayout_3.setStretch(1, 8)
        self.verticalLayout.addWidget(self.groupBox_2)
        self.groupBox_3 = QtWidgets.QGroupBox(self.verticalLayoutWidget)
        self.groupBox_3.setObjectName("groupBox_3")
        self.verticalLayoutWidget_7 = QtWidgets.QWidget(self.groupBox_3)
        self.verticalLayoutWidget_7.setGeometry(QtCore.QRect(0, 10, 421, 121))
        self.verticalLayoutWidget_7.setObjectName("verticalLayoutWidget_7")
        self.verticalLayout_7 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_7)
        self.verticalLayout_7.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_7.setObjectName("verticalLayout_7")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget_7)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_6.addWidget(self.label_3)
        self.SizeX_labe = QtWidgets.QLabel(self.verticalLayoutWidget_7)
        self.SizeX_labe.setFocusPolicy(QtCore.Qt.NoFocus)
        self.SizeX_labe.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.SizeX_labe.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.SizeX_labe.setTextFormat(QtCore.Qt.AutoText)
        self.SizeX_labe.setIndent(-1)
        self.SizeX_labe.setObjectName("SizeX_labe")
        self.horizontalLayout_6.addWidget(self.SizeX_labe)
        self.SizesldX = QtWidgets.QSlider(self.verticalLayoutWidget_7)
        self.SizesldX.setMinimum(1)
        self.SizesldX.setMaximum(10)
        self.SizesldX.setSingleStep(1)
        self.SizesldX.setPageStep(1)
        self.SizesldX.setOrientation(QtCore.Qt.Horizontal)
        self.SizesldX.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.SizesldX.setObjectName("SizesldX")
        self.horizontalLayout_6.addWidget(self.SizesldX)
        self.horizontalLayout_6.setStretch(1, 1)
        self.horizontalLayout_6.setStretch(2, 6)
        self.verticalLayout_7.addLayout(self.horizontalLayout_6)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget_7)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_5.addWidget(self.label_2)
        self.SizeY_labe = QtWidgets.QLabel(self.verticalLayoutWidget_7)
        self.SizeY_labe.setFocusPolicy(QtCore.Qt.NoFocus)
        self.SizeY_labe.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.SizeY_labe.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.SizeY_labe.setTextFormat(QtCore.Qt.AutoText)
        self.SizeY_labe.setIndent(-1)
        self.SizeY_labe.setObjectName("SizeY_labe")
        self.horizontalLayout_5.addWidget(self.SizeY_labe)
        self.SizesldY = QtWidgets.QSlider(self.verticalLayoutWidget_7)
        self.SizesldY.setMinimum(1)
        self.SizesldY.setMaximum(10)
        self.SizesldY.setSingleStep(1)
        self.SizesldY.setPageStep(1)
        self.SizesldY.setOrientation(QtCore.Qt.Horizontal)
        self.SizesldY.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.SizesldY.setObjectName("SizesldY")
        self.horizontalLayout_5.addWidget(self.SizesldY)
        self.horizontalLayout_5.setStretch(1, 1)
        self.horizontalLayout_5.setStretch(2, 6)
        self.verticalLayout_7.addLayout(self.horizontalLayout_5)
        self.verticalLayout.addWidget(self.groupBox_3)
        self.verticalLayout.setStretch(0, 1)
        self.verticalLayout.setStretch(1, 2)
        self.tabWidget.addTab(self.tab, "")
        self.tab_6 = QtWidgets.QWidget()
        self.tab_6.setObjectName("tab_6")
        self.verticalLayoutWidget_3 = QtWidgets.QWidget(self.tab_6)
        self.verticalLayoutWidget_3.setGeometry(QtCore.QRect(10, 0, 431, 201))
        self.verticalLayoutWidget_3.setObjectName("verticalLayoutWidget_3")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_3)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.groupBox_5 = QtWidgets.QGroupBox(self.verticalLayoutWidget_3)
        self.groupBox_5.setObjectName("groupBox_5")
        self.horizontalLayoutWidget_7 = QtWidgets.QWidget(self.groupBox_5)
        self.horizontalLayoutWidget_7.setGeometry(QtCore.QRect(9, 10, 401, 41))
        self.horizontalLayoutWidget_7.setObjectName("horizontalLayoutWidget_7")
        self.horizontalLayout_10 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_7)
        self.horizontalLayout_10.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_10.setObjectName("horizontalLayout_10")
        self.updown_label = QtWidgets.QLabel(self.horizontalLayoutWidget_7)
        self.updown_label.setFocusPolicy(QtCore.Qt.NoFocus)
        self.updown_label.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.updown_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.updown_label.setTextFormat(QtCore.Qt.AutoText)
        self.updown_label.setIndent(-1)
        self.updown_label.setObjectName("updown_label")
        self.horizontalLayout_10.addWidget(self.updown_label)
        self.updown_sld = QtWidgets.QSlider(self.horizontalLayoutWidget_7)
        self.updown_sld.setMinimum(-100)
        self.updown_sld.setMaximum(100)
        self.updown_sld.setSingleStep(5)
        self.updown_sld.setProperty("value", 1)
        self.updown_sld.setOrientation(QtCore.Qt.Horizontal)
        self.updown_sld.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.updown_sld.setObjectName("updown_sld")
        self.horizontalLayout_10.addWidget(self.updown_sld)
        self.horizontalLayout_10.setStretch(0, 2)
        self.horizontalLayout_10.setStretch(1, 8)
        self.verticalLayout_3.addWidget(self.groupBox_5)
        self.groupBox_4 = QtWidgets.QGroupBox(self.verticalLayoutWidget_3)
        self.groupBox_4.setObjectName("groupBox_4")
        self.horizontalLayoutWidget_6 = QtWidgets.QWidget(self.groupBox_4)
        self.horizontalLayoutWidget_6.setGeometry(QtCore.QRect(9, 10, 401, 41))
        self.horizontalLayoutWidget_6.setObjectName("horizontalLayoutWidget_6")
        self.horizontalLayout_8 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_6)
        self.horizontalLayout_8.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_8.setObjectName("horizontalLayout_8")
        self.leftright_label = QtWidgets.QLabel(self.horizontalLayoutWidget_6)
        self.leftright_label.setFocusPolicy(QtCore.Qt.NoFocus)
        self.leftright_label.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.leftright_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.leftright_label.setTextFormat(QtCore.Qt.AutoText)
        self.leftright_label.setIndent(-1)
        self.leftright_label.setObjectName("leftright_label")
        self.horizontalLayout_8.addWidget(self.leftright_label)
        self.leftright_sld = QtWidgets.QSlider(self.horizontalLayoutWidget_6)
        self.leftright_sld.setMinimum(-100)
        self.leftright_sld.setMaximum(100)
        self.leftright_sld.setSingleStep(5)
        self.leftright_sld.setProperty("value", 1)
        self.leftright_sld.setOrientation(QtCore.Qt.Horizontal)
        self.leftright_sld.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.leftright_sld.setObjectName("leftright_sld")
        self.horizontalLayout_8.addWidget(self.leftright_sld)
        self.horizontalLayout_8.setStretch(0, 2)
        self.horizontalLayout_8.setStretch(1, 8)
        self.verticalLayout_3.addWidget(self.groupBox_4)
        self.tabWidget.addTab(self.tab_6, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.formLayoutWidget = QtWidgets.QWidget(self.tab_2)
        self.formLayoutWidget.setGeometry(QtCore.QRect(9, 9, 431, 191))
        self.formLayoutWidget.setObjectName("formLayoutWidget")
        self.formLayout = QtWidgets.QFormLayout(self.formLayoutWidget)
        self.formLayout.setContentsMargins(0, 0, 0, 0)
        self.formLayout.setObjectName("formLayout")
        self.Gray_radioButton = QtWidgets.QRadioButton(self.formLayoutWidget)
        self.Gray_radioButton.setObjectName("Gray_radioButton")
        self.formLayout.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.Gray_radioButton)
        self.Hsv_radioButton = QtWidgets.QRadioButton(self.formLayoutWidget)
        self.Hsv_radioButton.setObjectName("Hsv_radioButton")
        self.formLayout.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.Hsv_radioButton)
        self.Bgr_radioButton = QtWidgets.QRadioButton(self.formLayoutWidget)
        self.Bgr_radioButton.setObjectName("Bgr_radioButton")
        self.formLayout.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.Bgr_radioButton)
        self.tabWidget.addTab(self.tab_2, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.tab_3)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(0, 0, 441, 212))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout_2.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout()
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.horizontalLayout_11 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_11.setObjectName("horizontalLayout_11")
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label.setObjectName("label")
        self.horizontalLayout_11.addWidget(self.label)
        self.x1_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.x1_label.setFocusPolicy(QtCore.Qt.NoFocus)
        self.x1_label.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.x1_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.x1_label.setTextFormat(QtCore.Qt.AutoText)
        self.x1_label.setIndent(-1)
        self.x1_label.setObjectName("x1_label")
        self.horizontalLayout_11.addWidget(self.x1_label)
        self.x1sld = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.x1sld.setMinimum(-100)
        self.x1sld.setMaximum(100)
        self.x1sld.setSingleStep(5)
        self.x1sld.setOrientation(QtCore.Qt.Horizontal)
        self.x1sld.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.x1sld.setObjectName("x1sld")
        self.horizontalLayout_11.addWidget(self.x1sld)
        self.horizontalLayout_11.setStretch(1, 2)
        self.horizontalLayout_11.setStretch(2, 8)
        self.verticalLayout_4.addLayout(self.horizontalLayout_11)
        self.horizontalLayout_12 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_12.setObjectName("horizontalLayout_12")
        self.label_7 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label_7.setObjectName("label_7")
        self.horizontalLayout_12.addWidget(self.label_7)
        self.y1_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.y1_label.setFocusPolicy(QtCore.Qt.NoFocus)
        self.y1_label.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.y1_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.y1_label.setTextFormat(QtCore.Qt.AutoText)
        self.y1_label.setIndent(-1)
        self.y1_label.setObjectName("y1_label")
        self.horizontalLayout_12.addWidget(self.y1_label)
        self.y1sld = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.y1sld.setMinimum(-100)
        self.y1sld.setMaximum(100)
        self.y1sld.setSingleStep(5)
        self.y1sld.setOrientation(QtCore.Qt.Horizontal)
        self.y1sld.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.y1sld.setObjectName("y1sld")
        self.horizontalLayout_12.addWidget(self.y1sld)
        self.horizontalLayout_12.setStretch(1, 2)
        self.horizontalLayout_12.setStretch(2, 8)
        self.verticalLayout_4.addLayout(self.horizontalLayout_12)
        self.verticalLayout_2.addLayout(self.verticalLayout_4)
        self.verticalLayout_5 = QtWidgets.QVBoxLayout()
        self.verticalLayout_5.setObjectName("verticalLayout_5")
        self.horizontalLayout_13 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_13.setObjectName("horizontalLayout_13")
        self.label_9 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label_9.setObjectName("label_9")
        self.horizontalLayout_13.addWidget(self.label_9)
        self.x2_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.x2_label.setFocusPolicy(QtCore.Qt.NoFocus)
        self.x2_label.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.x2_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.x2_label.setTextFormat(QtCore.Qt.AutoText)
        self.x2_label.setIndent(-1)
        self.x2_label.setObjectName("x2_label")
        self.horizontalLayout_13.addWidget(self.x2_label)
        self.x2sld = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.x2sld.setMinimum(-100)
        self.x2sld.setMaximum(100)
        self.x2sld.setSingleStep(5)
        self.x2sld.setOrientation(QtCore.Qt.Horizontal)
        self.x2sld.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.x2sld.setObjectName("x2sld")
        self.horizontalLayout_13.addWidget(self.x2sld)
        self.horizontalLayout_13.setStretch(1, 2)
        self.horizontalLayout_13.setStretch(2, 8)
        self.verticalLayout_5.addLayout(self.horizontalLayout_13)
        self.horizontalLayout_14 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_14.setObjectName("horizontalLayout_14")
        self.label_11 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label_11.setObjectName("label_11")
        self.horizontalLayout_14.addWidget(self.label_11)
        self.y2_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.y2_label.setFocusPolicy(QtCore.Qt.NoFocus)
        self.y2_label.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.y2_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.y2_label.setTextFormat(QtCore.Qt.AutoText)
        self.y2_label.setIndent(-1)
        self.y2_label.setObjectName("y2_label")
        self.horizontalLayout_14.addWidget(self.y2_label)
        self.y2sld = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.y2sld.setMinimum(-100)
        self.y2sld.setMaximum(100)
        self.y2sld.setSingleStep(5)
        self.y2sld.setOrientation(QtCore.Qt.Horizontal)
        self.y2sld.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.y2sld.setObjectName("y2sld")
        self.horizontalLayout_14.addWidget(self.y2sld)
        self.horizontalLayout_14.setStretch(1, 2)
        self.horizontalLayout_14.setStretch(2, 8)
        self.verticalLayout_5.addLayout(self.horizontalLayout_14)
        self.verticalLayout_2.addLayout(self.verticalLayout_5)
        self.verticalLayout_6 = QtWidgets.QVBoxLayout()
        self.verticalLayout_6.setObjectName("verticalLayout_6")
        self.horizontalLayout_15 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_15.setObjectName("horizontalLayout_15")
        self.label_14 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label_14.setObjectName("label_14")
        self.horizontalLayout_15.addWidget(self.label_14)
        self.x3_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.x3_label.setFocusPolicy(QtCore.Qt.NoFocus)
        self.x3_label.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.x3_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.x3_label.setTextFormat(QtCore.Qt.AutoText)
        self.x3_label.setIndent(-1)
        self.x3_label.setObjectName("x3_label")
        self.horizontalLayout_15.addWidget(self.x3_label)
        self.x3sld = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.x3sld.setMinimum(-100)
        self.x3sld.setMaximum(100)
        self.x3sld.setSingleStep(5)
        self.x3sld.setOrientation(QtCore.Qt.Horizontal)
        self.x3sld.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.x3sld.setObjectName("x3sld")
        self.horizontalLayout_15.addWidget(self.x3sld)
        self.horizontalLayout_15.setStretch(1, 2)
        self.horizontalLayout_15.setStretch(2, 8)
        self.verticalLayout_6.addLayout(self.horizontalLayout_15)
        self.horizontalLayout_16 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_16.setObjectName("horizontalLayout_16")
        self.label_16 = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.label_16.setObjectName("label_16")
        self.horizontalLayout_16.addWidget(self.label_16)
        self.y3_label = QtWidgets.QLabel(self.verticalLayoutWidget_2)
        self.y3_label.setFocusPolicy(QtCore.Qt.NoFocus)
        self.y3_label.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.y3_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.y3_label.setTextFormat(QtCore.Qt.AutoText)
        self.y3_label.setIndent(-1)
        self.y3_label.setObjectName("y3_label")
        self.horizontalLayout_16.addWidget(self.y3_label)
        self.y3sld = QtWidgets.QSlider(self.verticalLayoutWidget_2)
        self.y3sld.setMinimum(-100)
        self.y3sld.setMaximum(100)
        self.y3sld.setSingleStep(5)
        self.y3sld.setOrientation(QtCore.Qt.Horizontal)
        self.y3sld.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.y3sld.setObjectName("y3sld")
        self.horizontalLayout_16.addWidget(self.y3sld)
        self.horizontalLayout_16.setStretch(1, 2)
        self.horizontalLayout_16.setStretch(2, 8)
        self.verticalLayout_6.addLayout(self.horizontalLayout_16)
        self.verticalLayout_2.addLayout(self.verticalLayout_6)
        self.tabWidget.addTab(self.tab_3, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(self.tab_5)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(20, 0, 401, 51))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")
        self.horizontalLayout_9 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.horizontalLayout_9.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_9.setObjectName("horizontalLayout_9")
        self.Thresholding_label = QtWidgets.QLabel(self.horizontalLayoutWidget_5)
        self.Thresholding_label.setFocusPolicy(QtCore.Qt.NoFocus)
        self.Thresholding_label.setContextMenuPolicy(QtCore.Qt.NoContextMenu)
        self.Thresholding_label.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.Thresholding_label.setTextFormat(QtCore.Qt.AutoText)
        self.Thresholding_label.setIndent(-1)
        self.Thresholding_label.setObjectName("Thresholding_label")
        self.horizontalLayout_9.addWidget(self.Thresholding_label)
        self.Thresholdingsld = QtWidgets.QSlider(self.horizontalLayoutWidget_5)
        self.Thresholdingsld.setMaximum(255)
        self.Thresholdingsld.setSingleStep(5)
        self.Thresholdingsld.setTracking(True)
        self.Thresholdingsld.setOrientation(QtCore.Qt.Horizontal)
        self.Thresholdingsld.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.Thresholdingsld.setTickInterval(0)
        self.Thresholdingsld.setObjectName("Thresholdingsld")
        self.horizontalLayout_9.addWidget(self.Thresholdingsld)
        self.horizontalLayout_9.setStretch(0, 2)
        self.horizontalLayout_9.setStretch(1, 8)
        self.tabWidget.addTab(self.tab_5, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.formLayoutWidget_2 = QtWidgets.QWidget(self.tab_4)
        self.formLayoutWidget_2.setGeometry(QtCore.QRect(0, 0, 454, 201))
        self.formLayoutWidget_2.setObjectName("formLayoutWidget_2")
        self.formLayout_2 = QtWidgets.QFormLayout(self.formLayoutWidget_2)
        self.formLayout_2.setContentsMargins(0, 0, 0, 0)
        self.formLayout_2.setObjectName("formLayout_2")
        self.MeanFiltering_radioButton = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.MeanFiltering_radioButton.setFont(font)
        self.MeanFiltering_radioButton.setObjectName("MeanFiltering_radioButton")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.MeanFiltering_radioButton)
        self.GaussianFiltering_radioButton = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.GaussianFiltering_radioButton.setFont(font)
        self.GaussianFiltering_radioButton.setObjectName("GaussianFiltering_radioButton")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.GaussianFiltering_radioButton)
        self.EmbossImage_radioButton = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.EmbossImage_radioButton.setFont(font)
        self.EmbossImage_radioButton.setObjectName("EmbossImage_radioButton")
        self.formLayout_2.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.EmbossImage_radioButton)
        self.BilateralFilter_radioButton = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.BilateralFilter_radioButton.setFont(font)
        self.BilateralFilter_radioButton.setObjectName("BilateralFilter_radioButton")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.BilateralFilter_radioButton)
        self.MedianBlur_radioButton = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.MedianBlur_radioButton.setFont(font)
        self.MedianBlur_radioButton.setObjectName("MedianBlur_radioButton")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.MedianBlur_radioButton)
        self.EdgeDetectionImage_radioButton = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.EdgeDetectionImage_radioButton.setFont(font)
        self.EdgeDetectionImage_radioButton.setObjectName("EdgeDetectionImage_radioButton")
        self.formLayout_2.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.EdgeDetectionImage_radioButton)
        self.SobelFilter_radioButton = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.SobelFilter_radioButton.setFont(font)
        self.SobelFilter_radioButton.setObjectName("SobelFilter_radioButton")
        self.formLayout_2.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.SobelFilter_radioButton)
        self.AddGaussianNoise_radioButton = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.AddGaussianNoise_radioButton.setFont(font)
        self.AddGaussianNoise_radioButton.setObjectName("AddGaussianNoise_radioButton")
        self.formLayout_2.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.AddGaussianNoise_radioButton)
        self.ResultImage_radioButton = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.ResultImage_radioButton.setFont(font)
        self.ResultImage_radioButton.setObjectName("ResultImage_radioButton")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.ResultImage_radioButton)
        self.LaplacianFilter_radioButton = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.LaplacianFilter_radioButton.setFont(font)
        self.LaplacianFilter_radioButton.setObjectName("LaplacianFilter_radioButton")
        self.formLayout_2.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.LaplacianFilter_radioButton)
        self.AveragingFilter_radioButton = QtWidgets.QRadioButton(self.formLayoutWidget_2)
        font = QtGui.QFont()
        font.setFamily("Arial")
        font.setPointSize(10)
        self.AveragingFilter_radioButton.setFont(font)
        self.AveragingFilter_radioButton.setObjectName("AveragingFilter_radioButton")
        self.formLayout_2.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.AveragingFilter_radioButton)
        self.tabWidget.addTab(self.tab_4, "")
        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(10, 10, 1021, 521))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.horizontalLayout_4.setContentsMargins(0, 0, 0, 0)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.OriginPicture = QtWidgets.QLabel(self.horizontalLayoutWidget_4)
        self.OriginPicture.setObjectName("OriginPicture")
        self.horizontalLayout_4.addWidget(self.OriginPicture)
        self.RevisePicture = QtWidgets.QLabel(self.horizontalLayoutWidget_4)
        self.RevisePicture.setObjectName("RevisePicture")
        self.horizontalLayout_4.addWidget(self.RevisePicture)
        self.horizontalLayoutWidget_4.raise_()
        self.tabWidget.raise_()
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1048, 21))
        self.menubar.setObjectName("menubar")
        self.menu = QtWidgets.QMenu(self.menubar)
        self.menu.setObjectName("menu")
        self.menu_2 = QtWidgets.QMenu(self.menubar)
        self.menu_2.setObjectName("menu_2")
        self.menu_Perspective_Transform = QtWidgets.QMenu(self.menubar)
        self.menu_Perspective_Transform.setObjectName("menu_Perspective_Transform")
        self.menuROTA = QtWidgets.QMenu(self.menu_Perspective_Transform)
        self.menuROTA.setObjectName("menuROTA")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionLoadpicture = QtWidgets.QAction(MainWindow)
        self.actionLoadpicture.setObjectName("actionLoadpicture")
        self.actionclose = QtWidgets.QAction(MainWindow)
        self.actionclose.setObjectName("actionclose")
        self.action_Perspective_Transform = QtWidgets.QAction(MainWindow)
        self.action_Perspective_Transform.setObjectName("action_Perspective_Transform")
        self.actionHsv = QtWidgets.QAction(MainWindow)
        self.actionHsv.setObjectName("actionHsv")
        self.actionGray = QtWidgets.QAction(MainWindow)
        self.actionGray.setObjectName("actionGray")
        self.actionBgr = QtWidgets.QAction(MainWindow)
        self.actionBgr.setObjectName("actionBgr")
        self.actionROI = QtWidgets.QAction(MainWindow)
        self.actionROI.setObjectName("actionROI")
        self.actionThresholding = QtWidgets.QAction(MainWindow)
        self.actionThresholding.setObjectName("actionThresholding")
        self.actionHistogram_Equalization = QtWidgets.QAction(MainWindow)
        self.actionHistogram_Equalization.setObjectName("actionHistogram_Equalization")
        self.action_Image_histogram = QtWidgets.QAction(MainWindow)
        self.action_Image_histogram.setObjectName("action_Image_histogram")
        self.actionleft = QtWidgets.QAction(MainWindow)
        self.actionleft.setObjectName("actionleft")
        self.actionright = QtWidgets.QAction(MainWindow)
        self.actionright.setObjectName("actionright")
        self.actionVertically = QtWidgets.QAction(MainWindow)
        self.actionVertically.setObjectName("actionVertically")
        self.actionHorizontal = QtWidgets.QAction(MainWindow)
        self.actionHorizontal.setObjectName("actionHorizontal")
        self.actionHistogram = QtWidgets.QAction(MainWindow)
        self.actionHistogram.setObjectName("actionHistogram")
        self.menu.addAction(self.actionLoadpicture)
        self.menu.addAction(self.actionclose)
        self.menu.addSeparator()
        self.menu_2.addAction(self.actionHsv)
        self.menu_2.addAction(self.actionGray)
        self.menu_2.addAction(self.actionBgr)
        self.menuROTA.addAction(self.actionleft)
        self.menuROTA.addAction(self.actionright)
        self.menuROTA.addAction(self.actionVertically)
        self.menuROTA.addAction(self.actionHorizontal)
        self.menu_Perspective_Transform.addAction(self.action_Perspective_Transform)
        self.menu_Perspective_Transform.addAction(self.actionROI)
        self.menu_Perspective_Transform.addAction(self.actionHistogram_Equalization)
        self.menu_Perspective_Transform.addAction(self.action_Image_histogram)
        self.menu_Perspective_Transform.addAction(self.menuROTA.menuAction())
        self.menubar.addAction(self.menu.menuAction())
        self.menubar.addAction(self.menu_2.menuAction())
        self.menubar.addAction(self.menu_Perspective_Transform.menuAction())

        self.retranslateUi(MainWindow)
        self.tabWidget.setCurrentIndex(1)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.groupBox_2.setTitle(_translate("MainWindow", "旋轉"))
        self.Rota_label.setText(_translate("MainWindow", "0"))
        self.groupBox_3.setTitle(_translate("MainWindow", "大小"))
        self.label_3.setText(_translate("MainWindow", "X:"))
        self.SizeX_labe.setText(_translate("MainWindow", "1"))
        self.label_2.setText(_translate("MainWindow", "Y:"))
        self.SizeY_labe.setText(_translate("MainWindow", "1"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab), _translate("MainWindow", "幾何空間"))
        self.groupBox_5.setTitle(_translate("MainWindow", "上下"))
        self.updown_label.setText(_translate("MainWindow", "0"))
        self.groupBox_4.setTitle(_translate("MainWindow", "左右"))
        self.leftright_label.setText(_translate("MainWindow", "0"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_6), _translate("MainWindow", "位移"))
        self.Gray_radioButton.setText(_translate("MainWindow", "Gray"))
        self.Hsv_radioButton.setText(_translate("MainWindow", "Hsv"))
        self.Bgr_radioButton.setText(_translate("MainWindow", "Bgr"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_2), _translate("MainWindow", "色彩空間"))
        self.label.setText(_translate("MainWindow", "X1："))
        self.x1_label.setText(_translate("MainWindow", "0"))
        self.label_7.setText(_translate("MainWindow", "Y1："))
        self.y1_label.setText(_translate("MainWindow", "0"))
        self.label_9.setText(_translate("MainWindow", "X2："))
        self.x2_label.setText(_translate("MainWindow", "0"))
        self.label_11.setText(_translate("MainWindow", "Y2："))
        self.y2_label.setText(_translate("MainWindow", "0"))
        self.label_14.setText(_translate("MainWindow", "X3："))
        self.x3_label.setText(_translate("MainWindow", "0"))
        self.label_16.setText(_translate("MainWindow", "Y3："))
        self.y3_label.setText(_translate("MainWindow", "0"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_3), _translate("MainWindow", "仿射轉換"))
        self.Thresholding_label.setText(_translate("MainWindow", "0"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_5), _translate("MainWindow", "Thresholding"))
        self.MeanFiltering_radioButton.setText(_translate("MainWindow", "均值濾波(Mean Filtering)"))
        self.GaussianFiltering_radioButton.setText(_translate("MainWindow", "高斯濾波(Gaussian Filtering)"))
        self.EmbossImage_radioButton.setText(_translate("MainWindow", "影像浮雕(Emboss Image)"))
        self.BilateralFilter_radioButton.setText(_translate("MainWindow", "雙邊濾波(Bilateral filter)"))
        self.MedianBlur_radioButton.setText(_translate("MainWindow", "中值濾波(MedianBlur)"))
        self.EdgeDetectionImage_radioButton.setText(_translate("MainWindow", "邊緣檢測(Edge Detection Image)"))
        self.SobelFilter_radioButton.setText(_translate("MainWindow", "索伯算子(Sobel filter)"))
        self.AddGaussianNoise_radioButton.setText(_translate("MainWindow", "增加高斯噪點(Add gaussian noise)"))
        self.ResultImage_radioButton.setText(_translate("MainWindow", "Result Image"))
        self.LaplacianFilter_radioButton.setText(_translate("MainWindow", "拉普拉斯算子(Laplacian filter)"))
        self.AveragingFilter_radioButton.setText(_translate("MainWindow", "平均濾波器(Averaging filter)"))
        self.tabWidget.setTabText(self.tabWidget.indexOf(self.tab_4), _translate("MainWindow", "濾波"))
        self.OriginPicture.setText(_translate("MainWindow", "原圖"))
        self.RevisePicture.setText(_translate("MainWindow", "修改圖"))
        self.menu.setTitle(_translate("MainWindow", "檔案"))
        self.menu_2.setTitle(_translate("MainWindow", "色彩空間"))
        self.menu_Perspective_Transform.setTitle(_translate("MainWindow", "功能"))
        self.menuROTA.setTitle(_translate("MainWindow", "翻轉"))
        self.actionLoadpicture.setText(_translate("MainWindow", "載入圖片"))
        self.actionclose.setText(_translate("MainWindow", "關閉"))
        self.action_Perspective_Transform.setText(_translate("MainWindow", "透視投影轉換(Perspective Transform)"))
        self.actionHsv.setText(_translate("MainWindow", "Hsv"))
        self.actionGray.setText(_translate("MainWindow", "Gray"))
        self.actionBgr.setText(_translate("MainWindow", "Bgr"))
        self.actionROI.setText(_translate("MainWindow", "ROI"))
        self.actionThresholding.setText(_translate("MainWindow", "Thresholding"))
        self.actionHistogram_Equalization.setText(_translate("MainWindow", "Histogram Equalization"))
        self.action_Image_histogram.setText(_translate("MainWindow", "圖片直方圖(Image histogram)"))
        self.actionleft.setText(_translate("MainWindow", "左翻90度"))
        self.actionright.setText(_translate("MainWindow", "右翻90度"))
        self.actionVertically.setText(_translate("MainWindow", "水平翻轉"))
        self.actionHorizontal.setText(_translate("MainWindow", "垂直翻轉"))
        self.actionHistogram.setText(_translate("MainWindow", "Histogram"))

