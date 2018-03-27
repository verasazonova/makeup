import cv2
from face_annotator import FaceAnnotator
from face import Face
from makeup_applicator import apply_lipstick, apply_faded_lipstick, transfer_eyebrow
import numpy as np

eyebrow_template = cv2.imread('../templates/eyebrow_template.png')
eyebrow_line = np.array([[539,526], [599, 501], [663, 491], [728, 497], [781, 534]])

# loading the keypoint detection model, the image and the 3D model
predictor_path = "../shape_predictor_68_face_landmarks.dat"
images = ["adele.png", "input.jpg", "in3.png"]
face_annotator = FaceAnnotator(predictor_path)

# the smaller this value gets the faster the detection will work
# if it is too small, the user's face might not be detected

colors = [[26, 0, 69], [153, 88, 181], [158, 158, 247]]
alphas = [0.6, 0.2, 0.2]
fuzzes = [0.05, 0.1, 0.1]


for image_path in images:
    print(image_path)
    image_name = image_path[:-4]
    w = 300
    cameraImg = cv2.imread("../data/{}".format(image_path))
    h = int(w*cameraImg.shape[0]/cameraImg.shape[1])
    #image = cv2.resize(cameraImg, (w, h))
    image = cameraImg

    face_annotator.get_face_keypoints(image)

    face = Face(image)
    face.set_lips(*face_annotator.get_facepoint_lips())
    face.set_eyebrows(*face_annotator.get_eyebrows())

    face.interpolate_lips()
    face.resize_lips(1.1)
    name = '../results/{}_lipstick'.format(image_name)

    boundaries = face.get_lips_boundaries()

    for ind, (color, alpha, fuzz) in enumerate(zip(colors, alphas, fuzzes)):

        makeup_image = apply_lipstick(image, boundaries[0], color, fuzz)
        makeup_image = apply_lipstick(makeup_image, boundaries[1], color, fuzz)
        cv2.imwrite('{}_{}.png'.format(name, ind), makeup_image)

    brows = face.get_eyebrows()
    #makeup_image = transfer_eyebrow(image, brow[0], "../templates/eyebrow_template.jpg")

    print(brows[1])
    makeup_image = transfer_eyebrow(image, brows[1], eyebrow_template, eyebrow_line, image_name)


    cv2.imwrite('{}_brow.png'.format(name), makeup_image)

    # makeup_image = apply_faded_lipstick(image, lip_mask[0], [face_annotator.lower_lip_boundary], color, fuzz, alpha)
            # makeup_image = apply_faded_lipstick(makeup_image, lip_mask[1], [face_annotator.upper_lip_boundary], color, fuzz, alpha)
            # cv2.imwrite('{}_faded_{}.png'.format(name, ind), makeup_image)
