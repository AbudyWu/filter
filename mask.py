import cv2
from numba import jit
import numpy as np
from dft import dft,dft2D,idft,idft2D,dft_shift


@jit
def LPF(img,r):
    array = np.zeros((M,N),dtype=np.uint8)
    Mask = cv2.circle(array,(254,254),r,(1,1,1),-1)
    cv2.imshow("Mask:", np.uint8(Mask))
    imgaddmask= img * Mask
    return imgaddmask

@jit
def HPF(img,r):
    array = np.ones((M,N),dtype=np.uint8)
    # array2w = cv2.bitwise_not(array)
    Mask = cv2.circle(array,(254,254),r,(0,0,0),-1)
    cv2.imshow("Mask1:", np.uint8(Mask))
    imgaddmask =img * Mask
    return imgaddmask


if __name__ == "__main__":
    img = cv2.imread("Lenna.png")
    img2 = cv2.resize(img, (512, 512))
    img_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    M, N = img_gray.shape
    array2D = np.zeros((M, N), dtype=complex)
    dft1 = dft2D(img_gray,array2D)
    cv2.imshow("dft1:", np.uint8(dft1))

    dft1shift = dft_shift(dft1)
    lpf = LPF(dft1shift,100)
    cv2.imshow("lpf:",np.uint8(lpf))

    hpf = HPF(dft1shift,30)
    cv2.imshow("hpf:",np.uint8(hpf))

    lpfishift = dft_shift(lpf)
    cv2.imshow("lpfishift:",np.uint8(lpfishift))

    hpfishift = dft_shift(hpf)
    cv2.imshow("hpfishift:",np.uint8(hpfishift))

    idft1 = idft2D(lpfishift)
    cv2.imshow("idft1:", np.uint8(idft1))

    idft2 = idft2D(hpfishift)
    cv2.imshow("idft2:", np.uint8(np.abs(idft2)))



    cv2.waitKey(0)
