import cv2
from face_annotator import FaceAnnotator
from face import Face
from makeup_applicator import apply_lipstick, apply_faded_lipstick, transfer_eyebrow
import json
import requests

import logging
from logging.config import fileConfig

from flask import Flask, request
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger("tensorflow").setLevel(logging.WARNING)

app = Flask(__name__)


fileConfig('logging_config.ini')
logger = logging.getLogger()

face_annotator = None
net = None

predictor_path = "../shape_predictor_68_face_landmarks.dat"
w = 300


def download_image(url, path):
    logger.debug("got {}".format(url))
    response = requests.get(url)
    if response.status_code == 200:
        with open(path, 'wb') as f:
            f.write(response.content)
    logger.debug("Downoaded image from {}, to {}".format(url, path))


def log_json(label, payload):
    logger.debug( "=== %s ==="%label )
    logger.debug( json.dumps(payload, indent=2) )
    logger.debug( "================\n\n" )


@app.route('/makeup/health')
def health():
    return "Still alive!"


@app.route('/makeup/lipstick', methods=['POST', 'GET'])
def apply_lipstick():
    payload = json.loads(request.data)
    log_json("INCOMING", payload)

    url = payload["url"]
    path = "input_face.jpg"
    color = payload["color"]

    download_image(url, path)

    image = cv2.imread(path)

    #h = int(w*image.shape[0]/image.shape[1])
    #image = cv2.resize(cameraImg, (w, h))
    #image = cameraImg

    face_annotator.get_face_keypoints(image)

    face = Face(image)
    face.set_lips(*face_annotator.get_facepoint_lips())
    face.set_eyebrows(*face_annotator.get_eyebrows())

    face.interpolate_lips()
    face.resize_lips(1.1)

    boundaries = face.get_lips_boundaries()

    fuzz = 0.05

    makeup_image = apply_lipstick(image, boundaries[0], color, fuzz)
    makeup_image = apply_lipstick(makeup_image, boundaries[1], color, fuzz)
    cv2.imwrite('face.jpg', makeup_image)

    out = {"url": "/face.jpg"}

    log_json("OUTCOMING", out)
    return json.dumps(out)


if __name__ == '__main__':
    logger.debug("Initializing")
    face_annotator = FaceAnnotator(predictor_path)

    app.run(debug=True, host='0.0.0.0')
