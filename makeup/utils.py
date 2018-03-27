import cv2
import numpy as np
import scipy.interpolate as sc


def get_center(c):
    M = cv2.moments(c)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return cX, cY


def translate(img, dx, dy):
    num_rows, num_cols = img.shape[:2]
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(img, translation_matrix, (num_cols, num_rows))


def resize_keep_center(cnt, ratio):
    cx, cy = get_center(cnt)
    new_cnt = np.array([[int(x), int(y)] for x, y in ratio * cnt])

    cx_new, cy_new = get_center(new_cnt)
    new_cnt = new_cnt + [cx -cx_new, cy - cy_new]
    return new_cnt


def interpolate(points, k1='quadratic'):
    lx = points[:, 0]
    ly = points[:, 1]
    if lx[0] < lx[-1]:
        step = 1
    else:
        step = -1
    xnew = np.arange(lx[0], lx[-1] + step, step)
    f = sc.interp1d(lx, ly, kind=k1)
    ynew = f(xnew)
    new_points = np.array([[x, int(y)] for x, y in zip(xnew, ynew)])
    return new_points

