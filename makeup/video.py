import dlib
import cv2
import numpy as np

import models
import NonLinearLeastSquares
import ImageProcessing

from drawing import *

import FaceRendering
import utils

print "Press T to draw the keypoints and the 3D model"
print "Press R to start recording to a video file"


modelParams = None
lockedTranslation = False
drawOverlay = False
cap = cv2.VideoCapture(0)
writer = None
cameraImg = cap.read()[1]

textureImg = cv2.imread(image_name)
textureCoords = utils.getFaceTextureCoords(textureImg, mean3DShape, blendshapes, idxs2D, idxs3D, detector, predictor)
renderer = FaceRendering.FaceRenderer(cameraImg, textureImg, textureCoords, mesh)

goalLips, goal_pts1, goal_pts2 = getLips(textureImg, 'her.png')

#color = int(goalLips[goal_pts])
#print(color)
#print(idxs2D)

#print(idxs3D)

while True:
    cameraImg = cap.read()[1]
    shapes2D = utils.getFaceKeypoints(cameraImg, detector, predictor, maxImageSizeForDetection)

    if shapes2D is not None:
        for shape2D in shapes2D:
            #3D model parameter initialization
            modelParams = projectionModel.getInitialParameters(mean3DShape[:, idxs3D], shape2D[:, idxs2D])

            #3D model parameter optimization
            modelParams = NonLinearLeastSquares.GaussNewton(modelParams, projectionModel.residual, projectionModel.jacobian, ([mean3DShape[:, idxs3D], blendshapes[:, :, idxs3D]], shape2D[:, idxs2D]), verbose=0)

            #rendering the model to an image
            shape3D = utils.getShape3D(mean3DShape, blendshapes, modelParams)
            renderedImg = renderer.render(shape3D)

            if drawMode == 0:
                cameraImg = cameraImg

            elif drawMode == 1:
                # #blending of the rendered face with the image
                mask = np.copy(renderedImg[:, :, 0])
                cv2.imwrite('mask.png', mask)
                # print("Mask", mask.shape)
                renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask)
                cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)

            elif drawMode == 2:

                my_lips, my_pts = getLips(cameraImg, 'mine.png')

                # Find homography
                h, mask = cv2.findHomography(goal_pts, my_pts, cv2.RANSAC)
                 
                # Use homography
                height, width, channels = my_lips.shape
                goalLips_warped = cv2.warpPerspective(goalLips, h, (width, height))

                cv2.imwrite('warped.png', goalLips_warped)

                #try to mask only the mouth
                overlay = goalLips_warped.copy()
                # #blending of the rendered face with the image
                mask = np.copy(overlay[:, :, 0])
                # print("Mask", mask.shape)
                renderedImg = ImageProcessing.colorTransfer(cameraImg, overlay, mask)
                cameraImg = ImageProcessing.blendImages(renderedImg, cameraImg, mask)
           
                cv2.imwrite('result.png', cameraImg)

            elif drawMode == 3:
                
                cv2.imwrite('original.png', cameraImg)
                my_lips, my_pts1, my_pts2 = getLips(cameraImg, 'mine.png')

                # Find homography

                #try to mask only the mouth
                overlay = my_lips.copy()
                # color is in BGR
                cv2.fillPoly(overlay, [my_pts1, my_pts2], [33, 10, 168])
                alpha = 0.3
                cv2.addWeighted(my_lips, alpha, overlay, 1 - alpha, 0, overlay)
                cv2.imwrite('colored.png', overlay)

                mask = my_lips.copy()
                cv2.fillPoly(mask, [my_pts1, my_pts2], [255, 255, 255])

                cv2.imwrite('mask.png', mask)
                maskImg = cv2.imread("mask.png")

                # img = cv2.resize(maskImg, (0, 0), fx=1.03, fy=1.03)
                # cv2.imwrite('mask1.png', img)

                # pts = my_pts1 + my_pts2
                # # Find centroid of polygon
                # (meanx, meany) = pts.mean(axis=0)
                # print(meany, meany)

                # # Find centre of image
                # (cenx, ceny) = (img.shape[1]/2, img.shape[0]/2)

                # print(cenx, ceny)
                # # Make integer coordinates for each of the above
                # (meanx, meany, cenx, ceny) = np.floor([meanx, meany, cenx, ceny]).astype(np.int32)

                # # Calculate final offset to translate source pixels to centre of image
                # (offsetx, offsety) = (-meanx + cenx, -meany + ceny)

                # # Define remapping coordinates
                # (mx, my) = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
                # ox = (mx - offsetx).astype(np.float32)
                # oy = (my - offsety).astype(np.float32)

                # # Translate the image to centre
                # maskImg = cv2.remap(img, ox, oy, cv2.INTER_LINEAR)
                # cv2.imwrite('mask2.png', maskImg)

                mask = np.mean(maskImg, axis=2)

                # print("Mask", mask.shape)
                #renderedImg = ImageProcessing.colorTransfer(cameraImg, overlay, mask)
                #cv2.imwrite('rendered.png', renderedImg)    
                cameraImg = ImageProcessing.blendImages(overlay, cameraImg, mask)
                cv2.imwrite('result.png', cameraImg)

            else:

                #try to mask only the mouth

                pts = shape2D.T[48:60]
                mask1 = np.zeros((cameraImg.shape[0], cameraImg.shape[1]))
                overlay = cameraImg.copy()
                # color is in BGR !!!
                cv2.fillConvexPoly(overlay, pts, [0, 255, 0])
                xmin, ymin = np.min(pts, axis=0)
                xmax, ymax = np.max(pts, axis=0)
                cv2.fillConvexPoly(mask1, np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]]), 1)
                #print("Mask1", mask1.shape)

                # #blending of the rendered face with the image
                mask = np.copy(renderedImg[:, :, 0])
                # print("Mask", mask.shape)

                mask = np.mean(maskImg, axis=2)
                renderedImg = ImageProcessing.colorTransfer(cameraImg, renderedImg, mask1)
                overlay = ImageProcessing.blendImages(renderedImg, cameraImg, mask1)
                
                # color lips
                #cv2.fillConvexPoly(cameraImg, pts, [0, 255, 0])

                alpha = 0.8
                cv2.addWeighted(overlay, alpha, cameraImg, 1 - alpha, 0, cameraImg)



            #drawing of the mesh and keypoints
            if drawOverlay:
                #drawPoints(cameraImg, shape2D.T)
                #drawProjectedShape(cameraImg, [mean3DShape, blendshapes], projectionModel, mesh, modelParams, lockedTranslation)
                cv2.polylines(cameraImg,[my_pts1, my_pts2],True,(0,255,255))

    if writer is not None:
        writer.write(cameraImg)

    cv2.imshow('image', cameraImg)
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