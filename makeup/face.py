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


def resize_keep_center(img, ratio):

    cx, cy = get_center(cnt)

    mask = cv2.resize(img, (ratio, ratio))


class Face():
    def __init__(self, img):
        # the smaller this value gets the faster the detection will work
        # if it is too small, the user's face might not be detected

        self.image = img

        self.upper_inside_lip_boundary = None
        self.upper_outside_lip_boundary = None

        self.lower_inside_lip_boundary = None
        self.lower_outside_lip_boundary = None


