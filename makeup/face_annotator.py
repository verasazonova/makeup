import dlib
import cv2
import numpy as np


class FaceAnnotator():
    def __init__(self, predictor_path):
        # the smaller this value gets the faster the detection will work
        # if it is too small, the user's face might not be detected
        self.maxImageSizeForDetection = 640
    
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(predictor_path)
        self.face_points = None
        self.mouth_mask = None

        self.left_outside_corner = 48
        self.left_inside_corner = 60
        self.right_outside_corner = 54
        self.right_inside_corner = 64
        self.max = 68

        self.upper_lip_indx = [48, 49, 50, 51, 52, 53, 54, 64, 63, 62, 61, 60]
        self.lower_lip_indx = [48, 59, 58, 57, 56, 55, 54, 64, 65, 66, 67, 60]

        self.upper_outside_index = range(self.left_outside_corner, self.right_outside_corner +1 , 1)
        self.upper_inside_index = range(self.right_inside_corner, self.left_inside_corner -1 , -1)

        self.lower_outside_index = [self.left_outside_corner] + range(self.left_inside_corner -1, self.right_outside_corner -1, -1)
        self.lower_inside_index = range(self.right_inside_corner, self.max, 1) + [self.left_inside_corner]

        self.upper_lip_indx = self.upper_outside_index + self.upper_inside_index
        self.lower_lip_indx = self.lower_outside_index + self.lower_inside_index

        self.upper_lip_boundary = None
        self.lower_lip_boundary = None

        self.right_eyebrow_indx = range(22, 27)
        self.left_eyebrow_indx = range(17, 22)


    def get_face_keypoints(self, img):
        self.image = img

        imgScale = 1
        scaledImg = img
        if max(img.shape) > self.maxImageSizeForDetection:
            imgScale = self.maxImageSizeForDetection / float(max(img.shape))
            scaledImg = cv2.resize(img, (int(img.shape[1] * imgScale), int(img.shape[0] * imgScale)))

        dets = self.detector(scaledImg, 1)

        if len(dets) == 0:
            return None

        shapes2D = []
        for det in dets:
            faceRectangle = dlib.rectangle(int(det.left() / imgScale), int(det.top() / imgScale),
                                      int(det.right() / imgScale), int(det.bottom() / imgScale))

            dlibShape = self.predictor(img, faceRectangle)

            shape2D = np.array([[p.x, p.y] for p in dlibShape.parts()])
            shape2D = shape2D.T

            shapes2D.append(shape2D)

        self.face_points = shapes2D

    def get_facepoint_lips(self):
        return self.face_points[0].T[self.upper_outside_index], self.face_points[0].T[self.upper_inside_index], self.face_points[0].T[self.lower_outside_index], self.face_points[0].T[self.lower_inside_index]

    def get_eyebrows(self):
        return self.face_points[0].T[self.left_eyebrow_indx], self.face_points[0].T[self.right_eyebrow_indx]


def main():

    image_name = "../data/input.jpg"
    predictor_path = "../shape_predictor_68_face_landmarks.dat"
    face_annotator = FaceAnnotator(predictor_path)

    image = cv2.imread("../data/{}".format(image_name))

    face_annotator.analyze_face(image)

    lip_mask = face_annotator.mouth_mask

    #save_mask(image, lip_mask, '../results/mask.jpg')

    face_annotator.get_lips1()

    #green = np.uint8([[[0, 0, 255]]])
    #hsv_green = cv2.cvtColor(green, cv2.COLOR_BGR2HSV)
    #print(hsv_green)

if __name__ == "__main__":
    main()