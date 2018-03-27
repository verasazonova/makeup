import numpy as np
from skimage import color as cl
import ImageProcessing
import cv2
from utils import resize_keep_center, get_center


def add_moustage(frame, imgMustache, face_pos, nose_pos):

    #imgMustache = cv2.imread('mustache.png', -1)

    # Create the mask for the mustache
    orig_mask = imgMustache[:, :, 3]

    # Create the inverted mask for the mustache
    orig_mask_inv = cv2.bitwise_not(orig_mask)

    # Convert mustache image to BGR
    # and save the original image size (used later when re-sizing the image)
    imgMustache = imgMustache[:, :, 0:3]
    origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]

    # -----------------------------------------------------------------------------
    #       Main program loop
    # -----------------------------------------------------------------------------

    # collect video input from first webcam on system
    #video_capture = cv2.VideoCapture(0)

    # Capture video feed
    #ret, frame = video_capture.read()

    # Create greyscale image from the video feed
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in input video stream
    #faces = faceCascade.detectMultiScale(
    #    gray,
    #    scaleFactor=1.1,
    #    minNeighbors=5,
    #    minSize=(30, 30),
    #    flags=cv2.cv.CV_HAAR_SCALE_IMAGE
    #)

    x, y, w, h = face_pos

    roi_gray = gray[y:y + h, x:x + w]
    roi_color = frame[y:y + h, x:x + w]

    nx, ny, nw, nh = nose_pos

    # Detect a nose within the region bounded by each face (the ROI)
    #nose = noseCascade.detectMultiScale(roi_gray)

    # Un-comment the next line for debug (draw box around the nose)
    # cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)

    # The mustache should be three times the width of the nose
    mustacheWidth = 3 * nw
    mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth

    # Center the mustache on the bottom of the nose
    x1 = nx - (mustacheWidth / 4)
    x2 = nx + nw + (mustacheWidth / 4)
    y1 = ny + nh - (mustacheHeight / 2)
    y2 = ny + nh + (mustacheHeight / 2)

    # Check for clipping
    if x1 < 0:
        x1 = 0
    if y1 < 0:
        y1 = 0
    if x2 > w:
        x2 = w
    if y2 > h:
        y2 = h

    # Re-calculate the width and height of the mustache image
    mustacheWidth = x2 - x1
    mustacheHeight = y2 - y1

    # Re-size the original image and the masks to the mustache sizes
    # calcualted above
    mustache = cv2.resize(imgMustache, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(orig_mask, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)
    mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth, mustacheHeight), interpolation=cv2.INTER_AREA)

    # take ROI for mustache from background equal to size of mustache image
    roi = roi_color[y1:y2, x1:x2]

    # roi_bg contains the original image only where the mustache is not
    # in the region that is the size of the mustache.
    roi_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # roi_fg contains the image of the mustache only where the mustache is
    roi_fg = cv2.bitwise_and(mustache, mustache, mask=mask)

    # join the roi_bg and roi_fg
    dst = cv2.add(roi_bg, roi_fg)

    # place the joined image, saved to dst back over the original image
    roi_color[y1:y2, x1:x2] = dst


def overlay_image_alpha(img, img_overlay, pos, alpha_mask):
    """Overlay img_overlay on top of img at the position specified by
    pos and blend using alpha_mask.

    Alpha mask must contain values within the range [0, 1] and be the
    same size as img_overlay.
    """

    x, y = pos

    # Image ranges
    y1, y2 = max(0, y), min(img.shape[0], y + img_overlay.shape[0])
    x1, x2 = max(0, x), min(img.shape[1], x + img_overlay.shape[1])

    # Overlay ranges
    y1o, y2o = max(0, -y), min(img_overlay.shape[0], img.shape[0] - y)
    x1o, x2o = max(0, -x), min(img_overlay.shape[1], img.shape[1] - x)

    # Exit if nothing to do
    if y1 >= y2 or x1 >= x2 or y1o >= y2o or x1o >= x2o:
        return

    channels = img.shape[2]

    alpha = alpha_mask[y1o:y2o, x1o:x2o]
    alpha_inv = 1.0 - alpha

    for c in range(channels):
        img[y1:y2, x1:x2, c] = (alpha * img_overlay[y1o:y2o, x1o:x2o, c] +
                                alpha_inv * img[y1:y2, x1:x2, c])


def mask_from_color(img):
    mask = np.mean(img, axis=2)
    mask[np.where(mask > 0)] = 1
    return mask.astype(np.bool)


def mask_from_countour(points, w, h):
    mask = np.zeros((w, h))
    cv2.fillPoly(mask, [points], 1)
    return mask.astype(np.bool)


def save_mask(image, mask, name):
    out = np.zeros_like(image)
    out[mask] = image[mask]
    cv2.imwrite(name, out)


def get_bounding_rectangle(points, r):
    xmin, ymin = np.min(points, axis=0)
    xmax, ymax = np.max(points, axis=0)

    xmargin = int((r-1) * (xmax - xmin))
    ymargin = int((r-1) * (xmax - xmin))

    xmin = xmin - xmargin
    xmax = xmax + xmargin
    ymin = ymin - ymargin
    ymax = ymax + ymargin
    return np.array([[xmin, ymin], [xmin, ymax], [xmax, ymax], [xmax, ymin]])


def apply_lipstick(image, boundary, color, fuzz=0.1):
    im = image.copy()
    r, g, b = color  # lipstick color

    mask = mask_from_countour(boundary, im.shape[0], im.shape[1])

    n_points = np.count_nonzero(mask)
    val = cl.rgb2lab((im[mask] / 255.).reshape(n_points, 1, 3)).reshape(n_points, 3)
    L, A, B = np.mean(val[:, 0]), np.mean(val[:, 1]), np.mean(val[:, 2])
    L1, A1, B1 = cl.rgb2lab(np.array((r / 255., g / 255., b / 255.)).reshape(1, 1, 3)).reshape(3, )
    ll, aa, bb = L1 - L, A1 - A, B1 - B
    val[:, 0] += ll
    val[:, 1] += aa
    val[:, 2] += bb

    im[mask] = cl.lab2rgb(val.reshape(n_points, 1, 3)).reshape(n_points, 3) * 255
    return ImageProcessing.blendImages(im, image, mask, fuzz)


def apply_faded_lipstick(image, mask, points, color, alpha, fuzz=0.2):
    # try to mask only the mouth
    overlay = image.copy()
    # color is in BGR
    cv2.fillPoly(overlay, points, color)
    cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0, overlay)

    return ImageProcessing.blendImages(overlay, image, mask, fuzz)


def transfer_eyebrow(image, line, template, template_points, name):
    im = image.copy()

    r = 0.9

    line = resize_keep_center(line, r)

    homog, _ = cv2.findHomography(template_points, line, cv2.RANSAC)

    scaled_tmpl = cv2.warpPerspective(template, homog, (im.shape[0], im.shape[1]))
    template_mask = mask_from_color(scaled_tmpl)
    cv2.imwrite('../results/test_template.png', template)

    transfered = ImageProcessing.blendImages(scaled_tmpl, im, template_mask, featherAmount=0.1)
    cv2.imwrite('../results/{}.png'.format(name), transfered)

    return im
