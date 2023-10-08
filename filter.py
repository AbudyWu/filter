import cv2
import click
import numpy as np
from mkdir import mk
from matplotlib import pyplot as plt

mask_size = [3, 7, 15]
sobel_ksize = [3, 5, 7]
complex_ksize = [3,5]
sobel_ddepth = [cv2.CV_8U, cv2.CV_16S]
text = ["CV_8U", 0, 0, "CV_16S"]


def MeanFilter(img,path):
    cv2.imshow("ori", img)
    for i in mask_size:
        kernel = np.ones((i, i), dtype=np.float32) / (i * i)
        mean_img = cv2.filter2D(img, -1, kernel)
        cv2.imshow(f"mean{i}*{i}", mean_img)
        cv2.imwrite(f"./{path}/mean{i}*{i}.png",mean_img)
    cv2.waitKey(0)


def GaussianFilter(img,path):
    # img = cv2.imread(img, cv2.IMREAD_COLOR)
    cv2.imshow("ori", img)
    for i in mask_size:
        gaussian = cv2.GaussianBlur(img, (i, i), 7)
        cv2.imshow(f"gaussian{i}*{i}", gaussian)
        cv2.imwrite(f"./{path}/gaussian{i}*{i}.png", gaussian)
    cv2.waitKey(0)


def MedianFilter(img,path):
    # img = cv2.imread(img, cv2.IMREAD_COLOR)
    cv2.imshow("ori", img)
    for i in mask_size:
        median = cv2.medianBlur(img, i)
        cv2.imshow(f"median{i}*{i}", median)
        cv2.imwrite(f"./{path}/median{i}*{i}.png", median)
    cv2.waitKey(0)


def SobelFilter(img,path):
    # img = cv2.imread(img, cv2.IMREAD_COLOR)
    for j in sobel_ddepth:
        for i in sobel_ksize:
            x = cv2.Sobel(img, j, 1, 0, i)
            y = cv2.Sobel(img, j, 0, 1, i)
            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)
            dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            cv2.imshow(f"absX{i}*{i}_{text[j]}", absX)
            cv2.imshow(f"absY{i}*{i}_{text[j]}", absY)
            cv2.imshow(f"X+Y{i}*{i}_{text[j]}", dst)
            cv2.imwrite(f"./{path}/absX{i}*{i}_{text[j]}.png", absX)
            cv2.imwrite(f"./{path}/absY{i}*{i}_{text[j]}.png", absY)
            cv2.imwrite(f"./{path}/X+Y{i}*{i}_{text[j]}.png", dst)
            cv2.waitKey(0)


def CannyFilter(img,path):
    retval = cv2.useOptimized()
    cv2.setUseOptimized(True)
    print("Optimized",retval)
    # img = cv2.imread(img,0)
    for j in sobel_ddepth:
        for i in sobel_ksize:
            # x = cv2.Sobel(img, j, 1, 0, i,ddepth=cv2.CV_16S)
            # y = cv2.Sobel(img, j, 0, 1, i,ddepth=cv2.CV_16S)
            # absX = cv2.convertScaleAbs(x)
            # absY = cv2.convertScaleAbs(y)
            # canny = cv2.Canny(absX,absY,50,150)
            canny = cv2.Canny(img, j, 50, 150, apertureSize=i)
            cv2.imshow(f"Canny{i}*{i}_{text[j]}", canny)
            cv2.imwrite(f"./{path}/Canny{i}*{i}_{text[j]}.png", canny)
    cv2.waitKey(0)


def Complex(img,path):
    for i in complex_ksize:
        ##mean filter
        kernel = np.ones((5, 5), dtype=np.float32) / (5 * 5)
        mean_img = cv2.filter2D(img, -1, kernel)
        #sobel
        x = cv2.Sobel(mean_img, cv2.CV_16S, 1, 0, i)
        y = cv2.Sobel(mean_img, cv2.CV_16S, 0, 1, i)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        dst1 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        #canny
        canny1 = cv2.Canny(mean_img, cv2.CV_16S, 50, 150, apertureSize=i)
        ##show
        cv2.imshow(f"mean_sobel_x+y{i}*{i}",dst1)
        cv2.imshow(f"mean_sobel_canny{i}*{i}",canny1)
        cv2.imwrite(f"./{path}/mean_sobel_x+y{i}*{i}.png", dst1)
        cv2.imwrite(f"./{path}/mean_sobel_canny{i}*{i}.png", canny1)

        ##gaussian filter
        gaussian = cv2.GaussianBlur(img, (i, i), 7)
        #sobel
        xx = cv2.Sobel(gaussian, cv2.CV_16S, 1, 0, i)
        yy = cv2.Sobel(gaussian, cv2.CV_16S, 0, 1, i)
        absX = cv2.convertScaleAbs(xx)
        absY = cv2.convertScaleAbs(yy)
        dst2 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        #canny
        canny2 = cv2.Canny(gaussian, cv2.CV_16S, 50, 150, apertureSize=i)
        ##show
        cv2.imshow(f"gaussian_sobel_x+y{i}*{i}",dst2)
        cv2.imshow(f"gaussian_sobel_canny{i}*{i}",canny2)
        cv2.imwrite(f"./{path}/gaussian_sobel_x+y{i}*{i}.png", dst2)
        cv2.imwrite(f"./{path}/gaussian_sobel_canny{i}*{i}.png", canny2)
        cv2.waitKey(0)


if __name__ == "__main__":
    path1 = mk("filter_output", "smooth")
    path2 = mk("filter_output", "edge")
    path3 = mk("filter_output", "complex")

    Lenna = img = cv2.imread("Lenna.png",0)
    Lenna_color = img = cv2.imread("Lenna.png",cv2.IMREAD_COLOR)

    MeanFilter(Lenna,path1)
    GaussianFilter(Lenna,path1)
    MedianFilter(Lenna,path1)
    SobelFilter(Lenna,path2)
    CannyFilter(Lenna,path2)
    Complex(Lenna,path3)
