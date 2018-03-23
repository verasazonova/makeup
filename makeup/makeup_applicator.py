import numpy as np
from skimage import color as cl
import ImageProcessing
import cv2



def apply_lipstick(image, mask_orig, color, fuzz=0.1, boundary=None):
    im = image.copy()
    r, g, b = color  # lipstick color

    ratio = 1.01



    mask = mask.astype(np.bool)
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


