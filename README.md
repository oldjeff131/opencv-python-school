# **opencv-python-school**
* 使用opencv進行數位影像處理，使用的語言是python。
* UI介面使用Qtdesign進行設計。
---
## **B0**
> 把放在主程式的UI介面分離出來。
---
## **B0.5**
> 利用Qt Designer重新建立版面，重建一些code，減少重複的部分，舊程式碼與新UI介面的連結，重新命名主程式檔案。
---
## **B1**
> * UI:新增 morphological operations & cornerHarris 
> * 程式碼:新增 影像侵蝕(Erosion) & 影像膨脹(Dilation)
---
## 目前有的功能:
>> - [x] 載入圖片
> * 色彩空間
>> - [x] RGB
>> - [x] HSV
>> - [x] 灰階
> * 幾何轉換
>> - [x] 平移
>> - [x] 旋轉
>> - [x] 訪射轉換
>> - [x] 放大縮小
> * 影像資訊
>> - [x] 直方圖
>> - [ ] 影像大小
>> - [x] Histogram Equalization
> * 濾波
>> - [x] 均值濾波(Mean Filtering)
>> - [x] 影像浮雕(Emboss Image)
>> - [x] 高斯濾波(Gaussian Filtering)
>> - [x] 邊緣檢測(Edge Detection Image)
>> - [x] 中值濾波(MedianBlur)
>> - [x] 雙邊濾波(Bilateral filter)
>> - [x] 增加高斯噪點(Add gaussian noise)
>> - [x] Result Image
>> - [x] 拉普拉斯算子(Laplacian filter)
>> - [x] 平均濾波器(Averaging filter)
> * 其他功能
>> - [x] ROI
>> - [x] 透視投影轉換(Perspective Transform)
>> - [x] Erosion 影像侵蝕
>> - [x] Dilation 影像膨脹
>> - [ ] cornerHarris