from PyQt5.QtCore import *
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import *

class Ui_MainWindow(object):
    def intUI(self):#設定介面ui
        self.picturelabel = QLabel('picture',self)
        self.picturelabel.move(100,100)
        self.picturelabel.setGeometry(QRect(0, 0, 600, 400))

        self.picturelabe2 = QLabel('IeHmAction',self)
        self.picturelabe2.move(100,100)
        self.picturelabe2.setGeometry(QRect(300,100, 600, 400))

        self.picturelabe3 = QLabel('Thresholding',self)
        self.picturelabe3.move(100,100)
        self.picturelabe3.setGeometry(QRect(0, 350, 600, 400))

        self.picturelabe4 = QLabel('filter',self)
        self.picturelabe4.move(100,100)
        self.picturelabe4.setGeometry(QRect(300, 280, 600, 400))

        self.sld=QSlider(Qt.Horizontal,self)
        self.sld.setGeometry(50,600,150,50)
        self.sld.setMinimum(0)
        self.sld.setMaximum(255)
        self.sld.setTickPosition(QSlider.TicksRight)

        self.sldvaluelabel=QLabel("0",self)
        self.sldvaluelabel.move(100,100)
        self.sldvaluelabel.setGeometry(QRect(25, 600, 50, 50))

        self.sld1=QSlider(Qt.Horizontal,self)
        self.sld1.setGeometry(250,600,200,50)
        self.sld1.setMinimum(-360)
        self.sld1.setMaximum(360)
        self.sld1.setTickPosition(QSlider.TicksRight)

        self.sldvaluelabel1=QLabel("0",self)
        self.sldvaluelabel1.move(100,100)
        self.sldvaluelabel1.setGeometry(QRect(225, 600, 50, 50))

        self.Txvaluelabel1=QLabel("Tx:",self)
        self.Txvaluelabel1.move(20, 20)
        self.Txvaluelabel1.setGeometry(QRect(220, 670, 50, 25))

        self.Txtextbox = QLineEdit(self)
        self.Txtextbox.move(20, 20)
        self.Txtextbox.setGeometry(250,670,50,25)

        self.Tyvaluelabel1=QLabel("Ty:",self)
        self.Tyvaluelabel1.move(20, 20)
        self.Tyvaluelabel1.setGeometry(QRect(220, 700, 50, 25))

        self.Tytextbox = QLineEdit(self)
        self.Tytextbox.move(20, 20)
        self.Tytextbox.setGeometry(250,700,50,25)

        self.Sizevaluelabel1=QLabel("圖片縮放:",self)
        self.Sizevaluelabel1.move(20, 20)
        self.Sizevaluelabel1.setGeometry(QRect(25, 700, 50, 25))

        self.Sizextextbox = QLineEdit(self)
        self.Sizextextbox.move(20, 20)
        self.Sizextextbox.setGeometry(90,700,50,25)

        self.Sizeytextbox = QLineEdit(self)
        self.Sizeytextbox.move(20, 20)
        self.Sizeytextbox.setGeometry(150,700,50,25)

        self.affineluelabel1=QLabel("仿射位子\n上X下Y",self)
        self.affineluelabel1.move(20, 20)
        self.affineluelabel1.setGeometry(QRect(350, 670, 100, 50))

        self.affinex1textbox = QLineEdit(self)
        self.affinex1textbox.move(20, 20)
        self.affinex1textbox.setGeometry(400,670,50,25)

        self.affiney1textbox = QLineEdit(self)
        self.affiney1textbox.move(20, 20)
        self.affiney1textbox.setGeometry(400,700,50,25)

        self.affinex2textbox = QLineEdit(self)
        self.affinex2textbox.move(20, 20)
        self.affinex2textbox.setGeometry(460,670,50,25)

        self.affiney2textbox = QLineEdit(self)
        self.affiney2textbox.move(20, 20)
        self.affiney2textbox.setGeometry(460,700,50,25)

        self.affinex3textbox = QLineEdit(self)
        self.affinex3textbox.move(20, 20)
        self.affinex3textbox.setGeometry(520,670,50,25)

        self.affiney3textbox = QLineEdit(self)
        self.affiney3textbox.move(20, 20)
        self.affiney3textbox.setGeometry(520,700,50,25)

        layout = QGridLayout(self)
        layout.addWidget(self.picturelabel, 0, 0, 4, 4)
        layout.addWidget(self.picturelabe2, 0, 0, 4, 4)
        layout.addWidget(self.picturelabe3, 0, 0, 4, 4)

    def _createActions(self):#選單基礎設定
        self.OpenImageAction=QAction(self)
        self.OpenImageAction.setText("&開啟圖片\n(Open_Image)")
        self.ROIAction=QAction("&ROI",self)
        self.IeHmAction=QAction("&圖片直方圖(Image histogram)",self)
        self.grayAction=QAction("&Gray",self)
        self.hsvAction=QAction("&Hsv",self)
        #self.rgbAction=QAction("&Rgb",self)
        self.bgrAction=QAction("&Bgr",self)
        self.ThgAction=QAction("&Thresholding",self)
        self.HmEnAction=QAction("&Histogram Equalization",self)
        self.InfoAction=QAction("&影像資訊(Info)",self)
        self.FHAction=QAction("&垂直翻轉(Horizontal)",self)
        self.FVAction=QAction("&水平翻轉(Vertically)",self)
        self.FLAction=QAction("&向左翻轉(Left)",self)
        self.FRAction=QAction("&向右翻轉(Right)",self)
        self.ATAction=QAction("&仿射轉換(Affine)",self)
        self.MFAction=QAction("&均值濾波(Mean Filtering)",self)
        self.GFAction=QAction("&高斯濾波(Gaussian Filtering)",self)
        self.MBAction=QAction("&中值濾波(MedianBlur)",self)
        self.BFAction=QAction("&雙邊濾波(Bilateral filter)",self)
        #self.LPFAction=QAction("&低通濾波(Low-Pass Filter)",self)
        #self.HPFAction=QAction("&高通濾波(High-Pass Filter)",self)
        self.AGNFAction=QAction("&增加高斯噪點(Add gaussian noise)",self)
        self.SFAction=QAction("&索伯算子(Sobel filter)",self)
        self.LFAction=QAction("&拉普拉斯算子(Laplacian filter)",self)
        self.AFAction=QAction("&平均濾波器(Averaging filter)",self)
        self.ReloadAction=QAction("&重新載入(Reload)",self)
        self.TLAction=QAction("&平移(TransLation)",self)
        self.EDIction=QAction("&邊緣檢測(Edge Detection Image)",self)
        self.EIction=QAction("&影像浮雕(Emboss Image)",self)
        self.RIction=QAction("&Result Image",self)
        self.CSction=QAction("&改變大小",self)
        self.PTction=QAction("&透視投影轉換(Perspective Transform)",self)