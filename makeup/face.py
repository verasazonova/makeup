import numpy as np
from utils import resize_keep_center, interpolate

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
