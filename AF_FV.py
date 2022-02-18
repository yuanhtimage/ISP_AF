# from curses.ascii import FF
import cv2 as cv
import os
from cv2 import sqrt
import numpy as np
import matplotlib.pyplot as plt
import time

#   ROI处理
def ROI_Img(input_image,scale_x,scale_y):
    # 0 < scale_x,scale_y <0.5
    if (scale_x>0.5 or scale_y>0.5):
        print("Error parameter!")
        return 
    w, h =np.shape(input_image)
    result = img[int(scale_x*w):int((0.5+scale_x)*w), 
                    int(scale_y*h):int((0.5+scale_y)*h)]
    return result
#   数据归一化
def normalization(data):
    range = np.max(data) - np.min(data)
    return (data - np.min(data)) / range
#   直方图计算
def cala_hist(input_image):
    w, h = np.shape(input_image)
    count = np.zeros(256,np.float)
    for i in range(0,w):
        for j in range(0,h):
            pixel = input_image[i,j]
            index = int(pixel)
            count[index] = count[index]+1
    # 0~1概率版本
    # for i in range(0,256):
    #     count[i]= count[i]/(w*h)
    return count
#   能量梯度函数计算
def Energy_of_Gradient(input_image):
    """EOG: 能量梯度函数
    Args:
        input_image (narray): [输入图像]
    Returns:
        [float]: [清晰度评价值]
    """    
    w, h = np.shape(input_image)
    result = 0
    for x in range(0, w-1):
        for y in range(0, h-1):
            result+=(int(input_image[x+1,y])-int(input_image[x,y]))**2 
            + (int(input_image[x,y+1])-int(input_image[x,y]))**2 
    return result

#   Roberts函数计算
def Roberts_AF(input_image):
    w, h = np.shape(input_image)
    result = 0
    for x in range(0, w-1):
        for y in range(0, h-1):
            grad_x = int(input_image[x+1,y])-int(input_image[x,y])
            grad_y = int(input_image[x,y+1])-int(input_image[x,y])
            # result += sqrt(grad_x*grad_x +grad_y*grad_y)
            result += abs(grad_x) + abs(grad_y)
    return result

#   Tenengrad函数计算
def Tenengrad(input_image):
    # 方法1
    #分别求X,Y方向的梯度
    # grad_X=cv.Sobel(input_image,-1,1,0)
    # grad_Y=cv.Sobel(input_image,-1,0,1)
    # #求梯度图像
    # #grad=cv.addWeighted(grad_X,0.5,grad_Y,0.5,0)
    # #result=np.sum(grad**2)
    
    # 方法2
    grad_X = cv.Sobel(input_image, ddepth=cv.CV_64F, dx=1, dy=0, ksize=3)
    grad_Y = cv.Sobel(input_image, ddepth=cv.CV_64F, dx=0, dy=1, ksize=3)
    result = grad_X*grad_X + grad_Y*grad_Y
    result = cv.mean(result)[0]
    if np.isnan(result):
      return np.nanmean(result)
    return result

    # 方法3
    # w, h = np.shape(input_image)
    # result_image = np.zeros((w,h))
    # result_imageX = np.zeros((w,h))
    # result_imageY = np.zeros((w,h))
    # kernel_x = np.array([[-1,0,1],[-2,0,2],[-1,0,1]])
    # kernel_y = np.array([[1,2,1],[0,0,0],[-1,-2,-1]])  
    # for x in range(0, w-2):
    #     for y in range(0,h-2):
    #         result_imageX[x+1, y+1] = abs(np.sum(input_image[x:x+3, y:y+3]*kernel_x)) 
    #         result_imageY[x+1, y+1] = abs(np.sum(input_image[x:x+3, y:y+3]*kernel_y)) 
    #         result_image[x+1, y+1] = (result_imageX[x+1, y+1]*result_imageX[x+1,y+1] + 
    #                 result_imageY[x+1, y+1]*result_imageY[x+1, y+1])**0.5
    # result_image = np.uint8(result_image)
    # return result_image.var()

#   Brenner梯度函数计算
def Brenner(input_image):
    w, h = np.shape(input_image)
    result = 0
    for x in range(0, w-2):
        for y in range(0,h):
            result+=(int(input_image[x+2,y])-int(input_image[x,y]))**2
    return result

#   Variance方差函数计算
def Variance(input_image):
    result = 0
    # test = np.array(input_image)
    # result = test.var()
    u = np.mean(input_image)
    w, h = np.shape(input_image)
    for x in range(0,w):
        for y in range(0,h):
            result+=(input_image[x,y]-u)**2
    return result

#   Laplacian梯度函数计算
def Laplace(input_image):    
    # # 方法1
    return cv.Laplacian(input_image,cv.CV_64F).var()
    
    # 方法2
    # w, h = np.shape(input_image)
    # result_image = np.zeros((w, h))
    # kernel_laplace = np.array([[0,-1,0],[-1,4,-1],[0,-1,0]])
    # for x in range(0, w-2):
    #     for y in range(0,h-2):
    #         result_image[x+1, y+1] = abs(np.sum(input_image[x:x+3, y:y+3] * kernel_laplace))
    # result_image = np.uint8(result_image)
    # return result_image.var()

#   灰度方差函数计算
def SMD(input_image):
    w, h = np.shape(input_image)
    result = 0
    for x in range(0, w-1):
        for y in range(0, h):
            result+= np.fabs(int(input_image[x,y])-int(input_image[x,y-1]))
            result+= np.fabs(int(input_image[x,y])-int(input_image[x+1,y]))
    return result

#   灰度方差乘积计算
def SMD2(input_image):
    w, h = np.shape(input_image)
    result = 0
    for x in range(0, w-1):
        for y in range(0, h-1):
            result+=(np.fabs(int(input_image[x,y])-int(input_image[x+1,y]))*
                            np.fabs(int(input_image[x,y]-int(input_image[x,y+1])))
                            )
    return result

#   图像傅里叶变换
def fft(x):
    n = len(x)
    if n == 2:
        return [x[0] + x[1], x[0] - x[1]]
    
    G = fft(x[::2])
    H = fft(x[1::2])
    W = np.exp(-2j * np.pi * np.arange(n//2) / n)
    WH = W * H
    X = np.concatenate([G + WH, G - WH])
    return X
def fft2(img):
    h, w = img.shape
    if ((h-1) & h) or ((w-1) & w):
        print('Image size not a power of 2')
        return img
    
    img = normalization(img)
    res = np.zeros([h, w], 'complex128')
    for i in range(h):
        res[i, :] = fft(img[i, :])
    for j in range(w):
        res[:, j] = fft(res[:, j])
    return res
def fftshift(img):
    # swap the first and third quadrants, and the second and fourth quadrants
    h, w = img.shape
    h_mid, w_mid = h//2, w//2
    res = np.zeros([h, w], 'complex128')
    res[:h_mid, :w_mid] = img[h_mid:, w_mid:]
    res[:h_mid, w_mid:] = img[h_mid:, :w_mid]
    res[h_mid:, :w_mid] = img[:h_mid, w_mid:]
    res[h_mid:, w_mid:] = img[:h_mid, :w_mid]
    return np.abs(res)

#   离散傅里叶变换
def FFT(input_image):
    dft = cv.dft(np.float32(input_image), flags=cv.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)  # 进行数据重排 使低频信号位于中心处
    magnitude_spectrum = 20*np.log(cv.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
    # return magnitude_spectrum
    w, h = np.shape(input_image)
    result = 0
    for x in range(0, w-1):
        for y in range(0, h-1):
            result+=sqrt(x*x+y*y)*magnitude_spectrum[x,y]
    print(result[0]/(w*h))
    return result[0]/(w*h)
    # return magnitude_spectrum
    # fft2 = np.fft.fft2(input_image)
    # shift2center = np.fft.fftshift(fft2)
    # log_fft2 = np.log(1 + np.abs(fft2))
    # log_shift2center = np.log(1 + np.abs(shift2center))
    # return log_shift2center

#   离散余弦傅里叶变换
def DCT(input_image):
    img = input_image.astype('float')
    img_dct = cv.dct(img)
    img_dct_log = np.log(abs(img_dct))
    w, h = np.shape(input_image)
    result = 0
    for x in range(0, w-1):
        for y in range(0, h-1):
            result+=(x+y)*img_dct_log[x,y]
    print(result/(w*h))
    return result/(w*h)

#   信息熵函数计算
def Entropy(input_image):
    img_entropy = 0
    # hist_cv = cv.calcHist([input_image],[0],None,[256],[0,256])
    hist_cv = cala_hist(input_image)
    temp_p = hist_cv/(len(input_image)*len(input_image[0])) 
    for i in range(len(temp_p)):
        if (temp_p[i]== 0):
            img_entropy = img_entropy
        else:
            img_entropy = float(img_entropy - temp_p[i]*np.log2(1/temp_p[i]))
	# img_entropy = np.sum([p *np.log2(1/p) for p in P])
    return img_entropy

#   Vollaths图像相似性计算
def Vollaths(input_image):
    w, h = np.shape(input_image)
    u = np.mean(input_image)
    result = -w*h*(u**2)
    for x in range(0, w-1):
        for y in range(0, h):
            result+=int(input_image[x,y])*int(input_image[x+1,y])
    return result

#   Range灰度直方图计算
def Range(input_image):
    hist_cv = cv.calcHist([input_image],[0],None,[256],[0,256])
    # hist_cv = cala_hist(input_image)
    result = []
    for i in range(256):
        intensity = hist_cv[i]*i
        result.append(intensity)
    # return max(result)[0]-min(result)[0]
    return max(result)-min(result)

if __name__ == "__main__":
    start_time = time.time()
    
    # 基于图像梯度信息
    result_EOG = []
    result_Roberts = []
    result_Tenengrad = []
    result_Brenner = []
    result_Variance = []
    result_Laplace = []
    result_SMD = []
    result_SMD2 = []
    
    # 基于图像频域信息
    result_FFT = []
    result_DCT = []
    
    # 基于图像信息熵
    result_Entropy = []
    
    # 基于统计学图像信息
    result_Vollaths = []
    result_Range = []
    
    iter_num = 0
    #listdir的参数是文件夹的路径
    for filename in os.listdir(r"./input_image"):
        iter_num+=1             
        filenames = os.getcwd() +'/input_image/'+ filename
        img = cv.imread(filenames,0)
        img = ROI_Img(img,0.25,0.25)
        # result_EOG.append(Energy_of_Gradient(img))
        # result_Roberts.append(Roberts_AF(img))
        # result_Tenengrad.append(Tenengrad(img))
        # result_Brenner.append(Brenner(img))
        # result_Variance.append(Variance(img))
        # result_Laplace.append(Laplace(img))
        # result_SMD.append(SMD(img))
        # result_SMD2.append(SMD2(img))

        # result_FFT.append(FFT(img))
        # result_DCT.append(DCT(img))

        result_Entropy.append(Entropy(img))

        result_Vollaths.append(Vollaths(img))
        result_Range.append(Range(img))

        # cv.namedWindow('input_image',cv.WINDOW_NORMAL)
        # cv.imshow('input_image', img)
        # cv.waitKey(0)
    end_time = time.time()
    print("Run time...: " + str(round((end_time-start_time)*1000)) + "ms ")

    # plt.plot(np.arange(0,iter_num,1),normalization(result_EOG), color = 'r',label = 'EOG', marker = 'o')
    # plt.plot(np.arange(0,iter_num,1),normalization(result_Roberts), color = 'g', label = 'Roberts', marker = 'v') 
    # plt.plot(np.arange(0,iter_num,1),normalization(result_Tenengrad), color = 'b', label = 'Tenengrad', marker = 'd') 
    # plt.plot(np.arange(0,iter_num,1),normalization(result_Brenner), color = 'y',label = 'Brenner', marker = '*')
    # plt.plot(np.arange(0,iter_num,1),normalization(result_Variance), color = 'c',label = 'Variance', marker = 's')
    # plt.plot(np.arange(0,iter_num,1),normalization(result_Laplace), color = 'm',label = 'Laplace', marker = 'p')
    # plt.plot(np.arange(0,iter_num,1),normalization(result_SMD), color = 'k', label = 'SMD', marker = 'h') 
    # plt.plot(np.arange(0,iter_num,1),normalization(result_SMD2), color = 'pink', label = 'SMD2', marker = 'x') 
    
    # plt.plot(np.arange(0,iter_num,1),normalization(result_FFT), color = 'b',label = 'FFT', marker = 's')
    # plt.plot(np.arange(0,iter_num,1),normalization(result_DCT), color = 'r',label = 'DCT', marker = 'd')
    
    plt.plot(np.arange(0,iter_num,1),normalization(result_Entropy), color = 'g',label = 'Entropy', marker = 'o')
    
    plt.plot(np.arange(0,iter_num,1),normalization(result_Vollaths), color = 'b',label = 'Vollaths', marker = 's')
    plt.plot(np.arange(0,iter_num,1),normalization(result_Range), color = 'r',label = 'Range', marker = 'd')
    plt.legend()
    plt.xlabel('Num of images')
    plt.ylabel('AF value')
    plt.show()
