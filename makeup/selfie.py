import cv2
from face_annotator import FaceAnnotator
from makeup_applicator import apply_lipstick, apply_faded_lipstick


# loading the keypoint detection model, the image and the 3D model
predictor_path = "../shape_predictor_68_face_landmarks.dat"
images = ["input.jpg", "adele.png"]
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
    ratio = int(w*cameraImg.shape[0]/cameraImg.shape[1])
    image = cv2.resize(cameraImg, (w, ratio))

    #image = cameraImg
    print("Image: ", image.shape)

    for interp in [True, False]:

        face_annotator.analyze_face(image)
        name = '../results/{}_lipstick'.format(image_name)
        if interp:
            face_annotator.interpolate_lips()
            name = '../results/{}_lipstick_interpolated'.format(image_name)

        lip_mask = face_annotator.mouth_mask

        for ind, (color, alpha, fuzz) in enumerate(zip(colors, alphas, fuzzes)):
            makeup_image = apply_lipstick(image, lip_mask[0], color, fuzz, boundary=face_annotator.lower_lip_boundary)
            makeup_image = apply_lipstick(makeup_image, lip_mask[1], color, fuzz, boundary=face_annotator.upper_lip_boundary)
            cv2.imwrite('{}_{}.png'.format(name, ind), makeup_image)

            makeup_image = apply_faded_lipstick(image, lip_mask[0], [face_annotator.lower_lip_boundary], color, fuzz, alpha)
            makeup_image = apply_faded_lipstick(makeup_image, lip_mask[1], [face_annotator.upper_lip_boundary], color, fuzz, alpha)
            cv2.imwrite('{}_faded_{}.png'.format(name, ind), makeup_image)
