import dlib
import cv2
import numpy as np

import models
import NonLinearLeastSquares
import ImageProcessing

from drawing import *
from face_annotator import FaceAnnotator
from face import Face
from makeup_applicator import apply_lipstick, apply_faded_lipstick

import FaceRendering
import utils2

print "Press T to draw the keypoints and the 3D model"
print "Press R to start recording to a video file"

predictor_path = "../shape_predictor_68_face_landmarks.dat"
face_annotator = FaceAnnotator(predictor_path)
drawMode = 0

modelParams = None
lockedTranslation = False
drawOverlay = False
cap = cv2.VideoCapture(0)
writer = None
cameraImg = cap.read()[1]

color = [153, 88, 181]
fuzz = 0.1

while True:
    image = cap.read()[1]

    if drawMode == 1:
        # #blending of the rendered face with the image
        face_annotator.get_face_keypoints(image)
        face = Face(image)
        face.set_lips(*face_annotator.get_facepoint_lips())

        #face.interpolate_lips()
        #face.resize_lips(1.1)
        #name = '../results/{}_lipstick'.format(image_name)

        boundaries = face.get_lips_boundaries()
        makeup_image = apply_lipstick(image, boundaries[0], color, fuzz)
        makeup_image = apply_lipstick(makeup_image, boundaries[1], color, fuzz)

    else:
        makeup_image = cameraImg

    #drawing of the mesh and keypoints
    #if drawOverlay:
        #drawPoints(cameraImg, shape2D.T)
        #drawProjectedShape(cameraImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)
        #cv2.polylines(cameraImg,[my_pts1, my_pts2],True,(0,255,255))

    if writer is not None:
        writer.write(cameraImg)

    cv2.imshow('image', makeup_image)
    key = cv2.waitKey(1)

    if key == 27:
        break
    if key == ord('0'):
        drawMode = 0
    if key == ord('1'):
        drawMode = 1
    if key == ord('2'):
        drawMode = 2
    if key == ord('3'):
        drawMode = 3
    if key == ord('t'):
        drawOverlay = not drawOverlay
    if key == ord('r'):
        if writer is None:
            print "Starting video writer"
            writer = cv2.VideoWriter("../out.avi", cv2.cv.CV_FOURCC('X', 'V', 'I', 'D'), 25, (cameraImg.shape[1], cameraImg.shape[0]))

            if writer.isOpened():
                print "Writer succesfully opened"
            else:
                writer = None
                print "Writer opening failed"
        else:
            print "Stopping video writer"
            writer.release()
            writer = None