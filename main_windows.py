from fileinput import filename
import sys
import PyQt5
import click
import cv2 as cv
from cv2 import QT_CHECKBOX
import numpy as np
from matplotlib import image, pyplot as plt
from PyQt5.QtCore import *
from PyQt5.QtWidgets import * 
from PyQt5.QtGui import *
from pyrsistent import PTypeError
from scipy.misc import electrocardiogram
from scipy import ndimage
from sqlalchemy import true
import creat_ui

refPT=[] 
cropping = False
refPTx=[0,0,0,0]
refPTy=[0,0,0,0]
num=0
convolution=3

class Window(QMainWindow):
    def __init__(self,parent=None): #視窗建立
        super().__init__(parent)
        self.setWindowTitle("4a830212_opencv_homework")
        self.resize(1000,800)
        self.ui=creat_ui.Ui_MainWindow()
        self.ui.setupUi(self)
        self._connectActions()

    def _connectActions(self):#按鍵觸發
        self.buttongroup = QButtonGroup(self)
        self.buttongroup.addButton(self.ui.DilationradioButton)
        self.buttongroup.addButton(self.ui.ErosionradioButton)
        self.buttongroup2 = QButtonGroup(self)
        self.buttongroup2.addButton(self.ui.x3radioButton)
        self.buttongroup2.addButton(self.ui.x5radioButton)
        self.buttongroup2.addButton(self.ui.x7radioButton)
        self.ui.actionLoadpicture.triggered.connect(self.openSlot)
        self.ui.actioninfo.triggered.connect(self.pictureinfo)
        self.ui.actionROI.triggered.connect(self.Roi_control)
        self.ui.action_Image_histogram.triggered.connect(self.Histogram)
        self.ui.actionGray.triggered.connect(self.Gray_control)
        self.ui.Gray_radioButton.toggled.connect(self.Gray_control)
        self.ui.actionHsv.triggered.connect(self.Hsv_control)
        self.ui.Hsv_radioButton.toggled.connect(self.Hsv_control)
        #self.ui.rgbAction.triggered.connect(self.Rgb_control)
        self.ui.actionBgr.triggered.connect(self.Bgr_control)
        self.ui.Bgr_radioButton.toggled.connect(self.Bgr_control)
        self.ui.actionThresholding.triggered.connect(self.Thresholdingcontrol)
        self.ui.actionHistogram_Equalization.triggered.connect(self.Histogram_Equalization_control)
        self.ui.Thresholdingsld.valueChanged[int].connect(self.changeValue)
        self.ui.Rotasld.valueChanged[int].connect(self.changeValue)
        self.ui.SizesldY.valueChanged[int].connect(self.changeValue)
        self.ui.SizesldX.valueChanged[int].connect(self.changeValue)
        self.ui.x1sld.valueChanged[int].connect(self.changeValue)
        self.ui.y1sld.valueChanged[int].connect(self.changeValue)
        self.ui.x2sld.valueChanged[int].connect(self.changeValue)
        self.ui.y2sld.valueChanged[int].connect(self.changeValue)
        self.ui.x3sld.valueChanged[int].connect(self.changeValue)
        self.ui.y3sld.valueChanged[int].connect(self.changeValue)
        self.ui.updown_sld.valueChanged[int].connect(self.changeValue)
        self.ui.leftright_sld.valueChanged[int].connect(self.changeValue)
        self.ui.actionHorizontal.triggered.connect(self.pictureflip)
        self.ui.actionVertically.triggered.connect(self.pictureflip)
        self.ui.actionright.triggered.connect(self.pictureflip)
        self.ui.actionleft.triggered.connect(self.pictureflip)
        self.ui.MeanFiltering_radioButton.toggled.connect(self.Mean_Filtering)
        self.ui.GaussianFiltering_radioButton.toggled.connect(self.Gaussia_Filtering)
        self.ui.MedianBlur_radioButton.toggled.connect(self.MedianBlur)
        self.ui.BilateralFilter_radioButton.toggled.connect(self.Bilateral_filter)
        self.ui.AddGaussianNoise_radioButton.toggled.connect(self.add_gaussian_noise)
        self.ui.SobelFilter_radioButton.toggled.connect(self.sobel_filter)
        self.ui.LaplacianFilter_radioButton.toggled.connect(self.laplacian_filter)
        self.ui.AveragingFilter_radioButton.toggled.connect(self.averaging_filter)
        self.ui.EmbossImage_radioButton.toggled.connect(self.Emboss_Image)
        self.ui.EdgeDetectionImage_radioButton.toggled.connect(self.Edge_Detection_Image)
        self.ui.action_Perspective_Transform.triggered.connect(self.Perspective_transform)
        self.ui.ResultImage_radioButton.toggled.connect(self.Result_Image)
        self.ui.ErosionradioButton.toggled.connect(self.morphological_operations_onclick)
        self.ui.DilationradioButton.toggled.connect(self.morphological_operations_onclick)
        self.ui.x3radioButton.toggled.connect(self.convolution_onclick)
        self.ui.x5radioButton.toggled.connect(self.convolution_onclick)
        self.ui.x7radioButton.toggled.connect(self.convolution_onclick)
        # self.ui.cornerHarrissld.valueChanged.connect(self.cornerHarrissldchange)

    def openSlot(self): #載入的圖片
        filename, _ = QFileDialog.getOpenFileName(self, 'Open Image', 'Image', '*.png *.jpg *.bmp')
        if filename is '':
            return
        self.img = cv.imread(filename, -1)
        self.img_path=filename
        if self.img.size == 1:
            return
        self.showImage()

    def showImage(self): #顯示載入的圖片
        height, width, Channel = self.img.shape
        bytesPerline = 3 * width
        self.qImg = QImage(self.img.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.OriginPicture.setPixmap(QPixmap.fromImage(self.qImg))
        
    def Roi_control(self): #ROI
        img = cv.imread(self.img_path)
        roi = cv.selectROI(windowName="ROI", img=img, showCrosshair=False, fromCenter=False)
        x, y, w, h = roi
        cv.rectangle(img=img, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
        img_roi = img[int(y):int(y+h), int(x):int(x+w)]
        cv.imshow("roi", img)
        cv.imshow("roi_sel", img_roi)
        cv.waitKey(0)

    def Histogram(self): #影像直方圖
        img = cv.imread(self.img_path)
        plt.hist(img.ravel(), 256, [0, 256])
        plt.show()

    def Gray_control(self): #Gray
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        height, width = gray.shape
        bytesPerline = 1 * width
        self.qImg = QImage(gray, width, height, bytesPerline, QImage.Format_Grayscale8).rgbSwapped()
        self.ui.RevisePicture.setPixmap(QPixmap.fromImage(self.qImg))
        self.ui.RevisePicture.resize(self.qImg.size())
    
    def Bgr_control(self): #Bgr
        bgr = cv.cvtColor(self.img, cv.COLOR_RGB2BGR)
        height, width, channel = bgr.shape
        bytesPerline = 3 * width
        self.qImg = QImage(bgr, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.RevisePicture.setPixmap(QPixmap.fromImage(self.qImg))
        self.ui.RevisePicture.resize(self.qImg.size())

    def Hsv_control(self): #Hsv
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
        height, width, channel = hsv.shape
        bytesPerline = 3 * width
        self.qImg = QImage(hsv, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.RevisePicture.setPixmap(QPixmap.fromImage(self.qImg))
        self.ui.RevisePicture.resize(self.qImg.size())

    def Thresholdingcontrol(self):
        gray = cv.cvtColor(self.img, cv.COLOR_BGR2GRAY)
        ret, result = cv.threshold(gray, self.ui.Thresholdingsld.value(), 255, cv.THRESH_BINARY)
        height, width = result.shape
        bytesPerline = 1 * width
        self.qimg = QImage(result, width, height, bytesPerline, QImage.Format_Grayscale8).rgbSwapped()
        self.ui.RevisePicture.setPixmap(QPixmap.fromImage(self.qimg))
        self.ui.RevisePicture.resize(self.qimg.size())
    
    def Histogram_Equalization_control (self):
        img = cv.imread(self.img_path)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        cv.imshow('Original Image', gray)
        img_eq = cv.equalizeHist(gray)
        cv.imshow('Equalized Image', img_eq)
        plt.hist(gray.ravel(), 256, [0, 256])
        plt.hist(img_eq.ravel(), 256, [0, 256])
        plt.show()

    def changeValue(self,value):
        sender=self.sender()
        if sender==self.ui.Thresholdingsld:
            self.ui.Thresholdingsld.setValue(value)
            self.ui.Thresholding_label.setText(str(value))
            self.Thresholdingcontrol()
        elif sender==self.ui.Rotasld:
            self.ui.Rotasld.setValue(value)
            self.ui.Rota_label.setText(str(value))
            self.PictureRotaControl()
        elif sender==self.ui.SizesldX:
            self.ui.SizesldX.setValue(value)
            self.ui.SizeX_labe.setText(str(value))
            self.changesize()
        elif sender==self.ui.SizesldY:
            self.ui.SizesldY.setValue(value)
            self.ui.SizeY_labe.setText(str(value))
            self.changesize()
        elif sender==self.ui.x1sld:
            self.ui.x1sld.setValue(value)
            self.ui.x1_label.setText(str(value))
            self.AffineTransform()
        elif sender==self.ui.y1sld:
            self.ui.y1sld.setValue(value)
            self.ui.y1_label.setText(str(value))
            self.AffineTransform()
        elif sender==self.ui.x2sld:
            self.ui.x2sld.setValue(value)
            self.ui.x2_label.setText(str(value))
            self.AffineTransform()
        elif sender==self.ui.y2sld:
            self.ui.y2sld.setValue(value)
            self.ui.y2_label.setText(str(value))
            self.AffineTransform()
        elif sender==self.ui.x3sld:
            self.ui.x3sld.setValue(value)
            self.ui.x3_label.setText(str(value))
            self.AffineTransform()
        elif sender==self.ui.y3sld:
            self.ui.y3sld.setValue(value)
            self.ui.y3_label.setText(str(value))
            self.AffineTransform()
        elif sender==self.ui.updown_sld:
            self.ui.updown_sld.setValue(value)
            self.ui.updown_label.setText(str(value))
            self.PictureTranslation()
        elif sender==self.ui.leftright_sld:
            self.ui.leftright_sld.setValue(value)
            self.ui.leftright_label.setText(str(value))
            self.PictureTranslation()
        
    def pictureinfo(self):#圖片資訊
        img = cv.imread(self.img_path)
        size=img.shape
        QMessageBox.information(self,"Picture_info",str(size)+"\n(高度,寬度,像素)")

    def PictureRotaControl(self):#角度調整
        img = cv.imread(self.img_path)
        height, width, channel = img.shape
        center = (width // 2, height // 2)
        Pictureflip=cv.getRotationMatrix2D(center,self.ui.Rotasld.value(),1.0)
        Pictureflip = cv.warpAffine(img, Pictureflip, (width, height))
        bytesPerline = 3 * width
        Pictureflip = QImage(Pictureflip.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.RevisePicture.setPixmap(QPixmap.fromImage(Pictureflip))

    def pictureflip(self): #翻轉
        FlipBtn=self.sender()
        img = cv.imread(self.img_path)
        height, width, channel = img.shape
        if FlipBtn.text()=="垂直翻轉":
            Pictureflip = cv.flip(img, 0)
        elif FlipBtn.text()=="水平翻轉":
            Pictureflip = cv.flip(img, 1)
        else:
            center = (width // 2, height // 2)
            if FlipBtn.text()=="左翻90度":
                Pictureflip=cv.getRotationMatrix2D(center,-90,1.0)
            elif FlipBtn.text()=="右翻90度":
                Pictureflip = cv.getRotationMatrix2D(center,90,1.0)
            Pictureflip = cv.warpAffine(img, Pictureflip, (width, height))            
        bytesPerline = 3 * width
        Pictureflip = QImage(Pictureflip.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.RevisePicture.setPixmap(QPixmap.fromImage(Pictureflip))

    def PictureTranslation(self):#平移
        img = cv.imread(self.img_path)
        height, width, channel = img.shape
        rows, cols = img.shape[:2]
        affine = np.float32([[1, 0, int(self.ui.updown_label.text())], [0, 1, int(self.ui.leftright_label.text())]])
        dst = cv.warpAffine(img, affine, (cols, rows))
        bytesPerline = 3 * width
        Pictureflip = QImage(dst.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.RevisePicture.setPixmap(QPixmap.fromImage(Pictureflip))
    
    def changesize(self):
        img = cv.imread(self.img_path)
        rows, cols, ch = img.shape
        img_res = cv.resize(img, None, fx=(float(self.ui.SizeX_labe.text())), fy=(float(self.ui.SizeY_labe.text())), interpolation=cv.INTER_CUBIC)
        cv.imshow('resize image', img_res)

    def Mean_Filtering(self):#均值濾波 blur() boxFilter()
        img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_Mean=cv.blur(img_gray,(5,5))
        self.showpicturea(img_Mean,img)
        
    def Gaussia_Filtering(self):#高斯濾波
        img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_Gaussia=cv.GaussianBlur(img_gray,(11,11),-1)
        self.showpicturea(img_Gaussia,img)
    
    def MedianBlur(self):#中值濾波
        img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_median = cv.medianBlur(img_gray, 7)
        self.showpicturea(img_median,img)medianBlurcheckBox
    
    def Bilateral_filter(self):
        img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_Gaussia=cv.GaussianBlur(img_gray,(5,5),9)
        img_Bilateral=cv.bilateralFilter(img_Gaussia,10,10,10)
        self.showpicturea(img_Bilateral,img)

    def add_gaussian_noise(self):#增加高斯噪點
        img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img = img / 255
        mean = 0
        sigma = 0.2
        noise = np.random.normal(mean, sigma, img.shape)
        img_gaussian = img + noise
        img_gaussian = np.clip(img_gaussian, 0, 1)
        img_gaussian = np.uint8(img_gaussian * 255)
        noise = np.uint8(noise * 255)
        cv.imshow('Gaussian noise', noise)
        cv.imshow('noised image', img_gaussian)
        #median_filter(img_gaussian)
        img_result = cv.fastNlMeansDenoising(img_gaussian, None, 10, 10, 7)
        cv.imshow('fast denoise', img_result)

    def Emboss_Image(self): #影像浮雕
        img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kernel = np.array([[-2, -1, 0],[-1, 1, 1],[0, 1, 2]])
        img_result = cv.filter2D(img_gray, -1, kernel)
        self.showpicturea(img_result,img)

    def Edge_Detection_Image(self): #邊緣檢測
        img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kernel = np.array([[-1, -1, -1],[-1, 8, -1],[-1, -1, -1]])
        img_result = cv.filter2D(img_gray, -1, kernel)
        self.showpicturea(img_result,img)

    def Result_Image(self):
        img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        kernel = np.ones((7, 7), np.float32) / 49
        img_result = cv.filter2D(img_gray, -1, kernel)
        self.showpicturea(img_result,img)

    def sobel_filter(self):#索伯算子
        img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        x = cv.Sobel(img_gray, cv.CV_16S, 1, 0)
        y = cv.Sobel(img_gray, cv.CV_16S, 0, 1)
        abs_x = cv.convertScaleAbs(x)
        abs_y = cv.convertScaleAbs(y)
        img_sobel = cv.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
        cv.imshow('x-direction gradient image', abs_x)
        cv.imshow('y-direction gradient image', abs_y)
        cv.imshow('sobel image', img_sobel)

    def averaging_filter(self):#平均濾波器
        img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        img_averaging = cv.blur(img_gray, (5, 5))
        self.showpicturea(img_averaging,img)

    def laplacian_filter(self):#拉普拉斯算子
        img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
        img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        gray_lap = cv.Laplacian(img_gray, cv.CV_16S, ksize=5)
        img_laplacian = cv.convertScaleAbs(gray_lap)
        self.showpicturea(img_laplacian,img)

    def showpicturea(self,img,or_img):
        height, width, channel = or_img.shape
        bytesPerline = 1 * width
        img = QImage(img.data, width, height, bytesPerline, QImage.Format_Grayscale8).rgbSwapped()
        self.ui.RevisePicture.setPixmap(QPixmap.fromImage(img))

    def Erosion(self):
        global convolution
        img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
        kernel = np.ones((convolution,convolution), np.uint8)
        erosion = cv.erode(img, kernel, iterations = 1)
        cv.imshow('Erosion', erosion)
    
    def Dilation(self):
        global convolution
        img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
        kernel = np.ones((convolution,convolution), np.uint8)
        dilate = cv.dilate(img, kernel, iterations = 1)
        cv.imshow('Dilation', dilate)

    def AffineTransform(self):
        img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
        height, width, ch = img.shape
        pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
        pts2 = np.float32([[int(self.ui.x1_label.text()), int(self.ui.y1_label.text())], [int(self.ui.x2_label.text()), int(self.ui.y2_label.text())], [int(self.ui.x3_label.text()), int(self.ui.y3_label.text())]])
        M = cv.getAffineTransform(pts1, pts2)
        img_aff = cv.warpAffine(img, M, (width, height))
        bytesPerline = 3 * width
        Pictureflip = QImage(img_aff.data, width, height, bytesPerline, QImage.Format_RGB888).rgbSwapped()
        self.ui.RevisePicture.setPixmap(QPixmap.fromImage(Pictureflip))

    def OnMouseAction(self,event,x,y,flags,param):
        global refPT,cropping,num
        refPT=[(x,y)]
        if event==cv.EVENT_LBUTTONDOWN:
            refPT.append((x, y))
            if num<4:
                refPTx[num]=x
                refPTy[num]=y
            #print(str(refPT)+str(num)+' '+str(refPTx[num])+" "+str(refPTy[num]))
            num=num+1
            cropping = True  
        elif event == cv.EVENT_LBUTTONUP:
            cropping = False
            

    def Perspective_transform(self):
        global num
        img = cv.imread(self.img_path)
        cv.namedWindow("Perspective")
        cv.setMouseCallback("Perspective", self.OnMouseAction)
        while True:
            cv.imshow("Perspective",img)
            key = cv.waitKey(1) & 0xFF
            if key==ord("c"):
                break
            elif key==ord("r"):
                break
        pts1=np.float32([[refPTx[0],refPTy[0]],[refPTx[1],refPTy[1]],[refPTx[2],refPTy[2]],[refPTx[3],refPTy[3]]])
        pts2=np.float32([[0,0],[300,0],[300,300],[0,300]])
        M=cv.getPerspectiveTransform(pts1,pts2)
        dst=cv.warpPerspective(img,M,(300,300))
        cv.imshow('Perspective',dst)
        i=0
        while i<4:
            refPTx[i]=0
            refPTy[i]=0
            num=0
            i=i+1
                
    def convolution_onclick(self):
        global convolution
        radioBtn=self.sender()
        if radioBtn.isChecked():
            if radioBtn.text()=="3X3":
                convolution=3
            elif radioBtn.text()=="5X5":
                convolution=5
            elif radioBtn.text()=="7X7":
                convolution=7

    def morphological_operations_onclick(self):
        radioBtn=self.sender()
        if radioBtn.isChecked():
            if radioBtn.text()=="Dilation 影像膨脹":
                self.Dilation()
            elif radioBtn.text()=="Erosion 影像侵蝕":
                self.Erosion()
 
    # def cornerHarrissldchange(self,value):
    #     img = cv.imread(self.img_path,cv.COLOR_BGR2GRAY)
    #     gray=np.float32(img)
    #     self.ui.cornerHarrissld.setValue(value)
    #     self.ui.cornerHarris_labe.setText(str(value))
    #     dst=cv.cornerHarris(scr=gray,blockSize=5,ksize=7,k=0.04)
    #     a=dst>self.ui.cornerHarris_labe.text*dst.max()
    #     img[a]=[0,0,255]
    #     cv.imshow('corners')

if __name__=="__main__":
    app=QApplication(sys.argv)
    win=Window()
    win.show()
    sys.exit(app.exec_())