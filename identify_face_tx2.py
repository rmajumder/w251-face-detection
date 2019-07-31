import cv2
import time
import sys
import os
import numpy as np
import urllib
from PIL import Image
import tensorflow.contrib.tensorrt as trt
import matplotlib.pyplot as pyplot
import matplotlib.patches as patches
import tensorflow as tf
from scipy.spatial.distance import cosine
from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
import send_email_attach_aws_ses as sendemail

# extract a single face from a given photograph
def extract_face(filename, required_size=(224, 224)):
    # load image from file
    pixels = pyplot.imread(filename)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array


# extract faces and calculate face embeddings for a list of photo files
def get_embeddings(filenames):
    # extract faces
    faces = [extract_face(f) for f in filenames]
    # convert into an array of samples
    samples = asarray(faces, 'float32')
    # prepare the face for the model, e.g. center pixels
    samples = preprocess_input(samples, version=2)
    # create a vggface model
    model = VGGFace(model='resnet50', include_top=False, input_shape=(224, 224, 3), pooling='avg')
    # perform prediction
    yhat = model.predict(samples)
    return yhat

# determine if a candidate face is a match for a known face
def is_match(known_embedding, candidate_embedding, thresh=0.5):
    # calculate distance between embeddings
    score = cosine(known_embedding, candidate_embedding)
    matched = score <= thresh

    return matched, score, thresh


def print_match(isMatched):
    if isMatched[0]:
        print('>face is a Match (%.3f <= %.3f)' % (isMatched[1], isMatched[2]))
    else:
        print('>face is NOT a Match (%.3f > %.3f)' % (isMatched[1], isMatched[2]))

def get_resized_img(image):
    #plt.imshow(image)
    return np.array(image.resize((20, 20)))

def get_wm_image(cp):
    ret, frame = cp.read()
    timestamp = int(time.time())
    impath = "images-rishi/face-" + str(timestamp) + ".jpg"

    #image_resized = get_resized_img(frame)
    cv.imwrite(impath, frame)
    return impath

cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier('/opt/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
print("Loaded cascade classifier model")

rishi_img = ['images-rishi/rishi.jpeg']
rishi_240_img = ['images-rishi/rishi-240.jpg']
sharon_stone1_img = ['images-rishi/sharon_stone1.jpg']

rm_embed = get_embeddings(rishi_img)
rm_240_embed = get_embeddings(rishi_240_img)
ss_embed = get_embeddings(sharon_stone1_img)

while(True):
    ret, frame = cap.read()

    time.sleep(2)
    if ret == True:
        print("captured image")
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # face detection
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        for (x, y, w, h) in faces:
            rc, jpg = cv2.imencode('.png', gray_img)

            t0 = time.time()
            cur_img_path = get_wm_image(jpg)
            print(cur_img_path)
            unknown_embed = get_embeddings([cur_img_path])
            isMatched = is_match(rm_embed, unknown_embed)

            print_match(isMatched)
            t1 = time.time()

            if isMatched[0] == False:
                sendemail.send_email(cur_img_path, "Security Alert - Door 2", "Unidentified Person")
            t2 = time.time()

            print(f'Time to find match: {t1 - t0}')
            print(f'Time to find match and email it: {t2 - t0}')

            # os.remove(cur_img_path)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
