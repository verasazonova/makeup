import cv2
import numpy as np
import scipy.interpolate as sc
from utils import resize_keep_center, interpolate

# def get_lip_area_image(self):
#     mask = np.zeros((self.image.shape[0], self.image.shape[1]))
#
#     pts = np.vstack([self.upper_lip_boundary, self.lower_lip_boundary])
#     # rect = np.array([[ 660, 2052],[ 738 ,2073],[ 840, 2081], [ 925 ,2099]])
#     cv2.fillPoly(mask, [rect], 1)
#
#     mask = mask.astype(np.bool)
#     lip_area_image = np.zeros_like(self.image)
#     lip_area_image[mask] = self.image[mask]
#     return lip_area_image
#
#
# def get_lips1(self):
#     lip_area_image = self.get_lip_area_image()
#     cv2.imwrite('../results/lip_area.jpg', lip_area_image)
#
#     hsv_image = cv2.cvtColor(lip_area_image, cv2.COLOR_BGR2HSV)
#
#     lower_red = np.array([0, 100, 50])
#     upper_red = np.array([10, 255, 255])
#     mask1 = cv2.inRange(hsv_image, lower_red, upper_red)
#
#     lower_red = np.array([170, 100, 50])
#     upper_red = np.array([180, 255, 255])
#     mask2 = cv2.inRange(hsv_image, lower_red, upper_red)
#
#     mask = cv2.bitwise_or(mask1, mask2)
#     resut = cv2.bitwise_and(lip_area_image, lip_area_image, mask=mask)
#     # Bitwise-AND mask and original image
#
#     cv2.imwrite('../results/red_mask.jpg', mask)
#     cv2.imwrite('../results/masked_lip.jpg', resut)
#
#
# def get_lips(self):
#     lip_area_image = self.get_lip_area_image()
#     cv2.imwrite('../results/lip_area.jpg', lip_area_image)
#
#     hsv_image = cv2.cvtColor(lip_area_image, cv2.COLOR_BGR2HSV)
#
#     sobelX = cv2.Sobel(hsv_image, cv2.CV_64F, 1, 0)
#     sobelY = cv2.Sobel(hsv_image, cv2.CV_64F, 0, 1)
#     sobelCombined = cv2.bitwise_or(sobelX, sobelY)
#     cv2.imwrite('../results/gradient.png', sobelCombined)


class Face():
    def __init__(self, img):
        self.image = img
        self.upper_inside_lip_boundary = None
        self.upper_outside_lip_boundary = None
        self.lower_inside_lip_boundary = None
        self.lower_outside_lip_boundary = None
        self.left_eyebrow = None
        self.right_eyebrow = None

    def set_lips(self, uo, ui, lo, li):
        self.upper_inside_lip_boundary = ui
        self.upper_outside_lip_boundary = uo
        self.lower_inside_lip_boundary = li
        self.lower_outside_lip_boundary = lo

    def interpolate_lips(self):
        self.lower_outside_lip_boundary = interpolate(self.lower_outside_lip_boundary)
        self.lower_inside_lip_boundary = interpolate(self.lower_inside_lip_boundary)

        self.upper_inside_lip_boundary = interpolate(self.upper_inside_lip_boundary)
        self.upper_outside_lip_boundary = interpolate(self.upper_outside_lip_boundary, 'cubic')

    def get_lips_boundaries(self):
        pts1 = np.vstack([self.upper_outside_lip_boundary, self.upper_inside_lip_boundary])
        pts2 = np.vstack([self.lower_outside_lip_boundary, self.lower_inside_lip_boundary])
        return pts1, pts2

    def resize_lips(self, ratio):
        pts1 = np.vstack([self.upper_outside_lip_boundary, self.upper_inside_lip_boundary])
        new_pts1 = resize_keep_center(pts1, ratio)
        self.upper_outside_lip_boundary = new_pts1[:len(self.upper_outside_lip_boundary)]
        self.upper_inside_lip_boundary = new_pts1[len(self.upper_outside_lip_boundary):]

        pts1 = np.vstack([self.lower_outside_lip_boundary, self.lower_inside_lip_boundary])
        new_pts1 = resize_keep_center(pts1, ratio)
        self.lower_outside_lip_boundary = new_pts1[:len(self.lower_outside_lip_boundary)]
        self.lower_inside_lip_boundary = new_pts1[len(self.lower_inside_lip_boundary):]

    def set_eyebrows(self, leb, reb):
        self.left_eyebrow = leb
        self.right_eyebrow = reb

    def get_eyebrows(self):
        return self.left_eyebrow, self.right_eyebrow
