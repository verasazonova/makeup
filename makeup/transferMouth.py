import dlib
import cv2
import numpy as np

import models
import NonLinearLeastSquares
import ImageProcessing
from skimage.filters import sobel

from drawing import *

import FaceRendering
import utils


from pylab import *
from scipy.interpolate import interp1d
from skimage import color as cl



#you need to download shape_predictor_68_face_landmarks.dat from the link below and unpack it where the solution file is
#http://sourceforge.net/projects/dclib/files/dlib/v18.10/shape_predictor_68_face_landmarks.dat.bz2

#loading the keypoint detection model, the image and the 3D model
predictor_path = "../shape_predictor_68_face_landmarks.dat"
images = ["input"]
#the smaller this value gets the faster the detection will work
#if it is too small, the user's face might not be detected
maxImageSizeForDetection = 320

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

colors = [[26, 0, 69], [153, 88, 181], [158, 158, 247]]
alphas = [0.6, 0.2, 0.2]
fuzzes = [0.05, 0.1, 0.1]


def inter(lx, ly, k1='quadratic'):
    unew = np.arange(lx[0], lx[-1] + 1, 1)
    f2 = interp1d(lx, ly, kind=k1)
    return f2, unew

def apply_lipstick(im_name, points, color, name):

	points = np.array([[293, 580],
			[353, 581],
			[382, 587],
			[410, 578],
			[464, 584],
			[422, 637],
			[339, 637],
			[306, 583],
			[350, 588],
			[383, 595],
			[408, 587],
			[455, 585],
			[416, 615],
			[351, 619]])

	# gets the points on the boundary of lips from the file
	point_out_x = np.array((points[:len(points) // 2][:, 0]))
	point_out_y = np.array(points[:len(points) // 2][:, 1])
	point_in_x = (points[len(points) // 2:][:, 0])
	point_in_y = points[len(points) // 2:][:, 1]

	figure()
	im = imread(im_name)

	im = im.copy()
	r, g, b = color  # lipstick color

	up_left_end = 3
	up_right_end = 5


	# Code for the curves bounding the lips
	o_u_l = inter(point_out_x[:up_left_end], point_out_y[:up_left_end])
	o_u_r = inter(point_out_x[up_left_end - 1:up_right_end], point_out_y[up_left_end - 1:up_right_end])
	o_l = inter([point_out_x[0]] + point_out_x[up_right_end - 1:][::-1].tolist(),
	            [point_out_y[0]] + point_out_y[up_right_end - 1:][::-1].tolist(), 'cubic')
	i_u_l = inter(point_in_x[:up_left_end], point_in_y[:up_left_end])
	i_u_r = inter(point_in_x[up_left_end - 1:up_right_end], point_in_y[up_left_end - 1:up_right_end])
	i_l = inter([point_in_x[0]] + point_in_x[up_right_end - 1:][::-1].tolist(),
	            [point_in_y[0]] + point_in_y[up_right_end - 1:][::-1].tolist(), 'cubic')

	x = []  # will contain the x coordinates of points on lips
	y = []  # will contain the y coordinates of points on lips

	def ext(a, b, i):
	    a, b = np.round(a), np.round(b)
	    x.extend(arange(a, b, 1, dtype=np.int32).tolist())
	    y.extend((ones(int(b - a), dtype=np.int32) * i).tolist())


	for i in range(int(o_u_l[1][0]), int(i_u_l[1][0] + 1)):
	    ext(o_u_l[0](i), o_l[0](i) + 1, i)

	for i in range(int(i_u_l[1][0]), int(o_u_l[1][-1] + 1)):
	    ext(o_u_l[0](i), i_u_l[0](i) + 1, i)
	    ext(i_l[0](i), o_l[0](i) + 1, i)

	for i in range(int(i_u_r[1][-1]), int(o_u_r[1][-1] + 1)):
	    ext(o_u_r[0](i), o_l[0](i) + 1, i)

	for i in range(int(i_u_r[1][0]), int(i_u_r[1][-1] + 1)):
	    ext(o_u_r[0](i), i_u_r[0](i) + 1, i)
	    ext(i_l[0](i), o_l[0](i) + 1, i)


	# Now x and y contains coordinates of all the points on lips

	val = cl.rgb2lab((im[x, y] / 255.).reshape(len(x), 1, 3)).reshape(len(x), 3)
	L, A, B = mean(val[:, 0]), mean(val[:, 1]), mean(val[:, 2])
	L1, A1, B1 = cl.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3, )
	ll, aa, bb = L1 - L, A1 - A, B1 - B
	val[:, 0] += ll
	val[:, 1] += aa
	val[:, 2] += bb
	im.setflags(write=1)
	im[x, y] = cl.lab2rgb(val.reshape(len(x), 1, 3)).reshape(len(x), 3) * 255
	#gca().set_aspect('equal', adjustable='box')
	#imshow(im)
	#show()
	imsave(name, im)
	drawPoints(im, points)
	imsave('../created/with_mouth.jpg', im)



def get_edges(image, upper_lip, lower_lip, name):

	mask = np.zeros((image.shape[0], image.shape[1]))
	pts = np.vstack([upper_lip,lower_lip])
	xmin, ymin = np.min(pts, axis=0)
	xmax, ymax = np.max(pts, axis=0)

	xmargin = int(0.1*(xmax - xmin))
	ymargin = int(0.1*(xmax - xmin))

	xmin = xmin - xmargin
	xmax = xmax + xmargin
	ymin = ymin - ymargin
	ymax = ymax + ymargin
	rect = np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])
	#rect = np.array([[ 660, 2052],[ 738 ,2073],[ 840, 2081], [ 925 ,2099]])
	cv2.fillPoly(mask, [rect], 1)
	cv2.imwrite('../created/{}_box_mask.png'.format(name), mask)

	mask = mask.astype(np.bool)
	out = np.zeros_like(image)
	out += 1
	out[mask] = image[mask]    

	red_image = out[:, :, 1]	
	gray_image = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
	cv2.imwrite('../created/{}_box.png'.format(name), red_image)

	clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
	cl1 = clahe.apply(gray_image)
	drawPoints(cl1, np.array([[xmin +2, ymin+2], [xmax, ymin]]))
	cv2.imwrite('../created/{}_contrast.png'.format(name), cl1)
	
	basecolor = cl1[xmin, ymin]
	print(basecolor)

	leveled = cl1.copy()
	leveled[cl1 >= 190] = 0
	cv2.imwrite('../created/{}_leveled.png'.format(name), leveled)


	leveled = image.copy()
	leveled[gray_image >= 200] = 0
	cv2.imwrite('../created/{}_leveled2.png'.format(name), leveled)

	# #print(gray_image.shape)
	# #gradient = sobel(cl1)
	# sobelX = cv2.Sobel(cl1, cv2.CV_64F, 1, 0)
	# sobelY = cv2.Sobel(cl1, cv2.CV_64F, 0, 1)
	# sobelCombined = cv2.bitwise_or(sobelX, sobelY)
	# cv2.imwrite('../created/{}_gradient.png'.format(name), sobelCombined)

	#edges = cv2.Canny(sobelCombined, 100, 200)
	#cv2.imwrite('../created/{}_edges.png'.format(name), edges)

textureImg = cv2.imread('../data/photo-3.jpeg')
goalLips, goal_pts1, goal_pts2 = getLips(textureImg, '../created/her.png')

for image_name in images:

	cameraImg = cv2.imread("../data/{}.jpg".format(image_name))

	my_lips, my_pts1, my_pts2 = getLips(cameraImg, '../created/{}_mouth.png'.format(image_name))

	# Find homography
	h, mask = cv2.findHomography(goal_pts1, my_pts1, cv2.RANSAC)         
	# Use homography
	height, width, channels = my_lips.shape
	goalLips_warped1 = cv2.warpPerspective(goalLips, h, (width, height))
	transfered = ImageProcessing.blendImages(goalLips_warped1, cameraImg, mask)
	cv2.imwrite('../created/{}_warped.png'.format(image_name), goalLips_warped1)

	h, mask = cv2.findHomography(goal_pts2, my_pts2, cv2.RANSAC)         
	# Use homography
	height, width, channels = my_lips.shape
	goalLips_warped2 = cv2.warpPerspective(goalLips, h, (width, height))

	#cv2.imwrite('warped.png', goalLips_warped)
	transfered = ImageProcessing.blendImages(goalLips_warped2, transfered, mask)
	cv2.imwrite('../created/{}_transfered_result.png'.format(image_name), transfered)

	for ind, (color, alpha, fuzz) in enumerate(zip(colors, alphas, fuzzes)):

		#get_edges(cameraImg, my_pts1, my_pts2, '{}_{}'.format(image_name, ind))
		# Find homography

		apply_lipstick("../data/{}.jpg".format(image_name), my_pts1, color, '../created/{}_{}_applied.jpg'.format(image_name, ind))

		#try to mask only the mouth
		overlay = my_lips.copy()
		# color is in BGR
		cv2.fillPoly(overlay, [my_pts1, my_pts2], color)
		cv2.addWeighted(my_lips, 1 - alpha, overlay, alpha, 0, overlay)
		cv2.imwrite('../created/{}_colored.png'.format(image_name), overlay)

		mask = my_lips.copy()
		cv2.fillPoly(mask, [my_pts1, my_pts2], [255, 255, 255])

		cv2.imwrite('../created/mask.png', mask)
		maskImg = cv2.imread("../created/mask.png")

		mask = np.mean(maskImg, axis=2)

		colored = ImageProcessing.blendImages(overlay, cameraImg, mask, fuzz)
		cv2.imwrite('../created/{}_{}_result.png'.format(image_name, ind), colored)

		im = cameraImg.copy()
		r, g, b = color  # lipstick color

		mask = mask.astype(np.bool)
		n_points = np.count_nonzero(mask)
		val = cl.rgb2lab((im[mask] / 255.).reshape(n_points, 1, 3)).reshape(n_points, 3)
		L, A, B = mean(val[:, 0]), mean(val[:, 1]), mean(val[:, 2])
		L1, A1, B1 = cl.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3, )
		ll, aa, bb = L1 - L, A1 - A, B1 - B
		val[:, 0] += ll
		val[:, 1] += aa
		val[:, 2] += bb

		im[mask] = cl.lab2rgb(val.reshape(n_points, 1, 3)).reshape(n_points, 3) * 255
		colored = ImageProcessing.blendImages(im, cameraImg, mask, fuzz)		
		cv2.imwrite('../created/{}_{}_applied2.png'.format(image_name, ind), colored)

		im2 = cameraImg.copy()
		cv2.polylines(im2,[my_pts1, my_pts2],True,(0,255,255))
		drawPoints(im2, my_pts1 + my_pts2)
		cv2.imwrite('../created/{}_{}_result_with_mouth.png'.format(image_name, ind), im2)
