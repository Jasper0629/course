import cv2
import numpy as np
import math
import sys, time

img_2x = cv2.imread('depth_2x.png', cv2.IMREAD_GRAYSCALE)
img_4x = cv2.imread('depth_4x.png', cv2.IMREAD_GRAYSCALE)



def BiLinear_interpolation(img, dsth, dstw):
    scrH, scrW = img.shape
    scrH = scrH - 1
    scrW = scrW - 1
    print(img.shape)
    retimg = np.zeros((dsth, dstw), dtype=np.uint8)
    for i in range(dsth):
        for j in range(dstw):
            scrx = (i+0.5)*(scrH/dsth)-0.5
            scry = (j+0.5)*(scrW/dstw)-0.5
            x = math.floor(scrx)
            y = math.floor(scry)
            u = scrx-x
            v = scry-y
            retimg[i, j] = (1-u)*(1-v)*img[x, y]+u*(1-v)*img[x+1, y]+(1-u)*v*img[x, y+1]+u*v*img[x+1, y+1]
    return retimg


# Bicubic operation
def BiBubic(x):
    x = abs(x)
    if x <= 1:
        return 1 - 2 * (x ** 2) + (x ** 3)
    elif x < 2:
        return 4 - 8 * x + 5 * (x ** 2) - (x ** 3)
    else:
        return 0
# 双三次插值算法
# dstH为目标图像的高，dstW为目标图像的宽
def BiCubic_interpolation(img, dstH, dstW):
    scrH, scrW = img.shape
    # img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg = np.zeros((dstH, dstW, 3), dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx = i * (scrH / dstH)
            scry = j * (scrW / dstW)
            x = math.floor(scrx)
            y = math.floor(scry)
            u = scrx - x
            v = scry - y
            tmp = 0
            for ii in range(-1, 2):
                for jj in range(-1, 2):
                    if x + ii < 0 or y + jj < 0 or x + ii >= scrH or y + jj >= scrW:
                        continue
                    tmp += img[x + ii, y + jj] * BiBubic(ii - u) * BiBubic(jj - v)
            retimg[i, j] = np.clip(tmp, 0, 255)
    return retimg

if __name__ == '__main__':
    img2x_bilinear = BiLinear_interpolation(img_2x, 240, 320)
    img4x_bilinear = BiLinear_interpolation(img_4x, 240, 320)
    img2x_bicubic = BiCubic_interpolation(img_2x, 240, 320)
    img4x_bicubic = BiCubic_interpolation(img_4x, 240, 320)
    cv2.imwrite("img2x_bilinear.png", img2x_bilinear)
    cv2.imwrite("img4x_bilinear.png", img4x_bilinear)
    cv2.imwrite("img2x_bicubic.png", img2x_bicubic)
    cv2.imwrite("img4x_bicubic.png", img4x_bicubic)
    cv2.waitKey(0)