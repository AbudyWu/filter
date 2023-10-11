import cv2
import click
import numpy as np
from mkdir import mk
from matplotlib import pyplot as plt

mask_size = [3, 7, 15]
sobel_ksize = [3, 5, 7]
complex_ksize = [3, 5]
sobel_ddepth = [cv2.CV_8U, cv2.CV_16S]
text = ["CV_8U", 0, 0, "CV_16S"]


def MeanFilter(img, name, path):
    cv2.imshow(f"{name}_ori", img)
    for i in mask_size:
        kernel = np.ones((i, i), dtype=np.float32) / (i * i)
        mean_img = cv2.filter2D(img, -1, kernel)
        cv2.imshow(f"{name}_mean{i}*{i}", mean_img)
        cv2.imwrite(f"./{path}/{name}_mean{i}*{i}.png", mean_img)
    cv2.waitKey(0)


def GaussianFilter(img, name, path):
    cv2.imshow(f"{name}_ori", img)
    for i in mask_size:
        gaussian = cv2.GaussianBlur(img, (i, i), 7)
        cv2.imshow(f"{name}_gaussian{i}*{i}", gaussian)
        cv2.imwrite(f"./{path}/{name}_gaussian{i}*{i}.png", gaussian)
    cv2.waitKey(0)


def MedianFilter(img, name, path):
    cv2.imshow(f"{name}_ori", img)
    for i in mask_size:
        median = cv2.medianBlur(img,ksize= i)
        cv2.imshow(f"{name}_median{i}*{i}", median)
        cv2.imwrite(f"./{path}/{name}_median{i}*{i}.png", median)
    cv2.waitKey(0)


def SobelFilter(img, name, path):
    for j in sobel_ddepth:
        for i in sobel_ksize:
            x = cv2.Sobel(img, j, 1, 0, ksize= i)
            y = cv2.Sobel(img, j, 0, 1, ksize= i)
            absX = cv2.convertScaleAbs(x)
            absY = cv2.convertScaleAbs(y)
            dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
            cv2.imshow(f"{name}_absX{i}*{i}_{text[j]}", absX)
            cv2.imshow(f"{name}_absY{i}*{i}_{text[j]}", absY)
            cv2.imshow(f"{name}_X+Y{i}*{i}_{text[j]}", dst)
            cv2.imwrite(f"./{path}/{name}_absX{i}*{i}_{text[j]}.png", absX)
            cv2.imwrite(f"./{path}/{name}_absY{i}*{i}_{text[j]}.png", absY)
            cv2.imwrite(f"./{path}/{name}_X+Y{i}*{i}_{text[j]}.png", dst)
            cv2.waitKey(0)


def CannyFilter(img, name, path):
    retval = cv2.useOptimized()
    cv2.setUseOptimized(True)
    print("Optimized", retval)
    for j in sobel_ddepth:
        for i in sobel_ksize:
            canny = cv2.Canny(img, 50, 150, apertureSize=i)
            cv2.imshow(f"{name}_Canny{i}*{i}_{text[j]}", canny)
            cv2.imwrite(f"./{path}/{name}_Canny{i}*{i}_{text[j]}.png", canny)
    cv2.waitKey(0)


def Complex(img, name, path):
    for i in complex_ksize:
        ##mean filter
        kernel = np.ones((5, 5), dtype=np.float32) / (5 * 5)
        mean_img = cv2.filter2D(img, -1, kernel)
        # sobel
        x = cv2.Sobel(mean_img, cv2.CV_16S, 1, 0,ksize= i)
        y = cv2.Sobel(mean_img, cv2.CV_16S, 0, 1,ksize= i)
        absX = cv2.convertScaleAbs(x)
        absY = cv2.convertScaleAbs(y)
        dst1 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        # canny
        canny1 = cv2.Canny(mean_img, cv2.CV_16S, 50, 150, apertureSize=i)
        ##show
        cv2.imshow(f"{name}_mean_sobel_x+y{i}*{i}", dst1)
        cv2.imshow(f"{name}_mean_sobel_canny{i}*{i}", canny1)
        cv2.imwrite(f"./{path}/{name}_mean_sobel_x+y{i}*{i}.png", dst1)
        cv2.imwrite(f"./{path}/{name}_mean_sobel_canny{i}*{i}.png", canny1)

        ##gaussian filter
        gaussian = cv2.GaussianBlur(img, (i, i), 7)
        # sobel
        xx = cv2.Sobel(gaussian, cv2.CV_16S, 1, 0,ksize= i)
        yy = cv2.Sobel(gaussian, cv2.CV_16S, 0, 1,ksize= i)
        absX = cv2.convertScaleAbs(xx)
        absY = cv2.convertScaleAbs(yy)
        dst2 = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        # canny
        canny2 = cv2.Canny(gaussian, cv2.CV_16S, 50, 150, apertureSize=i)
        ##show
        cv2.imshow(f"{name}_gaussian_sobel_x+y{i}*{i}", dst2)
        cv2.imshow(f"{name}_gaussian_sobel_canny{i}*{i}", canny2)
        cv2.imwrite(f"./{path}/{name}_gaussian_sobel_x+y{i}*{i}.png", dst2)
        cv2.imwrite(f"./{path}/{name}_gaussian_sobel_canny{i}*{i}.png", canny2)
        cv2.waitKey(0)


if __name__ == "__main__":
    path1 = mk("filter_output", "smooth")
    path2 = mk("filter_output", "edge")
    path3 = mk("filter_output", "complex")
    # path4 = mk("filter_output","noise")

    Lenna = cv2.imread("Lenna.png", 0)
    # Lenna_color = cv2.imread("Lenna.png", cv2.IMREAD_COLOR)
    car_ori = cv2.imread("dgg.jpg", 0)
    babe_ori = cv2.imread("babe.jpg", 0)
    car = cv2.resize(car_ori, (512, 512))
    babe = cv2.resize(babe_ori, (512, 512))

    # star = cv2.imread("noise.bmp",0)
    # MedianFilter(star,"star",path4)

    MeanFilter(Lenna,"Lenna",path1)
    MeanFilter(babe, "babe", path1)
    MeanFilter(car, "car", path1)

    GaussianFilter(Lenna,"Lenna",path1)
    GaussianFilter(babe, "babe", path1)
    GaussianFilter(car, "car", path1)

    MedianFilter(Lenna,"Lenna",path1)
    MedianFilter(babe, "babe", path1)
    MedianFilter(car, "car", path1)

    SobelFilter(Lenna,"Lenna",path2)
    SobelFilter(babe,"babe",path2)
    SobelFilter(car,"car",path2)

    CannyFilter(Lenna,"Lenna",path2)
    CannyFilter(babe,"babe",path2)
    CannyFilter(car,"car",path2)

    Complex(Lenna,"Lenna",path3)
    Complex(babe,"babe",path3)
    Complex(car,"car",path3)
