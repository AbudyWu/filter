import cv2
import numpy as np
from decorator import CannySpeedUp
from matplotlib import pyplot as plt


class Smooth:
    def __init__(self) -> None:
        pass

    def MeanFilter(self, img, i, *args, **kwargs):
        self.kernel = np.ones((i, i), dtype=np.float32) / (i * i)
        self.mean_img = cv2.filter2D(img, -1, self.kernel)
        return self.mean_img

    def GaussianFilter(self, img, i, simX=5, *args, **kwargs):
        self.gaussian = cv2.GaussianBlur(img, (i, i), simX)
        return self.gaussian

    def MedianFilter(self, img, i, *args, **kwargs):
        self.median = cv2.medianBlur(img, i)
        return self.median


class Edge:
    def __init__(self) -> None:
        pass

    def SobelFilter(self, img, i, ddepth=cv2.CV_8U, *args, **kwargs):
        self.x = cv2.Sobel(self, img, ddepth, 1, 0, ksize=i)
        self.y = cv2.Sobel(self, img, ddepth, 0, 1, ksize=i)
        self.absX = cv2.convertScaleAbs(self, x)
        self.absY = cv2.convertScaleAbs(self, y)
        self.dst = cv2.addWeighted(self, absX, 0.5, absY, 0.5, 0)
        return self.absX, self.absY, self.dst

    @CannySpeedUp
    def CannyFilter(self, img, i, threshold1=50, threshold2=150, *args, **kwargs):
        self.canny = cv2.Canny(self, img, threshold1, threshold2, apertureSize=i)
        return self.canny


class Complex(Smooth, Edge):
    def __init__(self):
        self.ddepth = cv2.CV_16S

    def mean_sobel(self, img, i, *args, **kwargs):
        mean_img = Smooth.MeanFilter(img, i)
        return Edge.SobelFilter(mean_img, i, Complex.ddepth)

    def mean_canny(self, img, i, *args, **kwargs):
        mean_img = Smooth.MeanFilter(img, i)
        return Edge.CannyFilter(
            mean_img, self.ddepth, threshold1=50, threshold2=150, apertureSize=i
        )

    def gaussian_sobel(self, img, i, *args, **kwargs):
        gaussian_img = Smooth.GaussianFilter(img, i, simX=7)
        return Edge.SobelFilter(gaussian_img, i, Complex.ddepth)

    def gaussian_canny(self, img, i, *args, **kwargs):
        gaussian_img = Smooth.GaussianFilter(img, i, simX=7)
        return Edge.CannyFilter(
            gaussian_img, self.ddepth, threshold1=50, threshold2=150, apertureSize=i
        )


if __name__ == "__main__":
    img = cv2.imread("babe.jpg", cv2.IMREAD_COLOR)
    filter_ls = [Smooth.MeanFilter, Smooth.GaussianFilter, Smooth.MedianFilter]
    filter_name_ls = ["Mean", "Gaussian", "Median"]
    class_ls = [Smooth]
    fig, axs = plt.subplots(3, 3, figsize=(18, 18))

    kernel_size_ls = [3, 15, 27]

    for i, (filter_func, filter_name) in enumerate(zip(filter_ls, filter_name_ls)):
        for j, (kernel_size) in enumerate(kernel_size_ls):
            result = filter_func(self=class_ls[0], img=img, i=kernel_size)
            axs[i, j].imshow(
                cv2.cvtColor(result, cv2.COLOR_BGR2RGB),
                extent=[0, result.shape[1], result.shape[0], 0],
            )
            axs[i, j].set_title(f"{filter_name} (ksize {kernel_size})")
            axs[i, j].axis("off")

    plt.tight_layout()
    plt.show()
