import dlib
import cv2
import numpy as np
import scipy.interpolate as sc


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

    def get_face_keypoints(self, img):
        self.image = img
        print('Set image', self.image.shape)

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
        self.lower_lip_boundary = self.face_points[0].T[self.lower_lip_indx]
        self.upper_lip_boundary = self.face_points[0].T[self.upper_lip_indx]
        # create mask
        self.mouth_mask = []
        for points in [self.lower_lip_boundary, self.upper_lip_boundary]:
            mask = np.zeros((self.image.shape[0], self.image.shape[1]))
            cv2.fillPoly(mask, [points], 1)
            mask = mask.astype(np.bool)
            self.mouth_mask.append(mask)
        print(self.mouth_mask[0].shape)


    def analyze_face(self, image):
        self.get_face_keypoints(image)
        self.get_facepoint_lips()

    def get_lip_area_image(self):
        mask = np.zeros((self.image.shape[0], self.image.shape[1]))

        pts = np.vstack([self.upper_lip_boundary, self.lower_lip_boundary])
        xmin, ymin = np.min(pts, axis=0)
        xmax, ymax = np.max(pts, axis=0)

        xmargin = int(0.1 * (xmax - xmin))
        ymargin = int(0.1 * (xmax - xmin))

        xmin = xmin - xmargin
        xmax = xmax + xmargin
        ymin = ymin - ymargin
        ymax = ymax + ymargin
        rect = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
        # rect = np.array([[ 660, 2052],[ 738 ,2073],[ 840, 2081], [ 925 ,2099]])
        cv2.fillPoly(mask, [rect], 1)

        mask = mask.astype(np.bool)
        lip_area_image = np.zeros_like(self.image)
        lip_area_image[mask] = self.image[mask]
        return lip_area_image

    def get_lips1(self):
        lip_area_image = self.get_lip_area_image()
        cv2.imwrite('../results/lip_area.jpg', lip_area_image)

        hsv_image = cv2.cvtColor(lip_area_image, cv2.COLOR_BGR2HSV)

        lower_red = np.array([0, 100, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv_image, lower_red, upper_red)

        lower_red = np.array([170, 100, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv_image, lower_red, upper_red)

        mask = cv2.bitwise_or(mask1, mask2)
        resut = cv2.bitwise_and(lip_area_image, lip_area_image, mask=mask)
        # Bitwise-AND mask and original image

        cv2.imwrite('../results/red_mask.jpg', mask)
        cv2.imwrite('../results/masked_lip.jpg', resut)

    def get_lips(self):

        lip_area_image = self.get_lip_area_image()
        cv2.imwrite('../results/lip_area.jpg', lip_area_image)

        hsv_image = cv2.cvtColor(lip_area_image, cv2.COLOR_BGR2HSV)

        sobelX = cv2.Sobel(hsv_image, cv2.CV_64F, 1, 0)
        sobelY = cv2.Sobel(hsv_image, cv2.CV_64F, 0, 1)
        sobelCombined = cv2.bitwise_or(sobelX, sobelY)
        cv2.imwrite('../results/gradient.png', sobelCombined)


    def interpolate_lips(self):
        boundary = self.face_points[0].T[self.lower_outside_index]
        lo_points = interpolate(boundary)

        boundary = self.face_points[0].T[self.lower_inside_index]
        li_points = interpolate(boundary)
        self.lower_lip_boundary = np.vstack((lo_points, li_points))

        boundary = self.face_points[0].T[self.upper_outside_index]
        uo_points = interpolate(boundary, 'cubic')

        boundary = self.face_points[0].T[self.upper_inside_index]
        ui_points = interpolate(boundary)
        self.upper_lip_boundary = np.vstack((uo_points, ui_points))


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


def save_mask(image, mask, name):
    out = np.zeros_like(image)
    out[mask] = image[mask]
    cv2.imwrite(name, out)


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