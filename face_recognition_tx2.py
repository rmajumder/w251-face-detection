import cv2
import time
import sys
import os
import numpy as np
import urllib
from PIL import Image
import tensorflow.contrib.tensorrt as trt
import face_recognition
# import matplotlib.pyplot as pyplot
# import matplotlib.patches as patches
# import tensorflow as tf
# from scipy.spatial.distance import cosine
# from mtcnn.mtcnn import MTCNN
# from keras_vggface.vggface import VGGFace
# from keras_vggface.utils import preprocess_input
import send_email_attach_aws_ses as sendemail

# Our list of known face encodings and a matching list of metadata about each face.
known_face_encodings = []

def load_known_faces():
    global known_face_encodings, known_face_metadata

    files = glob.glob('known-images/*.jpg')
    for f in files:
        frame = cv2.imread(f)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        face_locations = face_recognition.face_locations(small_frame)
        face_encodings = face_recognition.face_encodings(small_frame, face_locations)
        known_face_encodings.append(face_encodings[0])

def lookup_known_face(face_encoding):
    if len(known_face_encodings) == 0:
        return None

    # Calculate the face distance between the unknown face and every face in our known face list
    # This will return a floating point number between 0.0 and 1.0 for each known face. The smaller the number,
    # the more similar that face was to the unknown face.
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

    # Get the known face that had the lowest distance (i.e. most similar) from the unknown face.
    best_match_index = np.argmin(face_distances)

    return face_distances[best_match_index]

def check_if_matched(score, threshold=0.5):
    if score <= threshold:
        return True
    else:
        return False

cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier('/opt/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
print("Loaded cascade classifier model")

load_known_faces()
print("Loaded known faces encodings")


while(True):
    ret, frame = cap.read()

    time.sleep(2)
    if ret == True:
        print("captured image")
        # face detection
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        for (x, y, w, h) in faces:
            t0 = time.time()
            # cur_img_path = get_wm_image(frame)
            # print(cur_img_path)
            # unknown_embed = get_embeddings([cur_img_path])
            # isMatched = is_match(rm_embed, unknown_embed)

            # print_match(isMatched)
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            score = lookup_known_face(face_encodings[0])

            t1 = time.time()
            is_matched = check_if_matched(score)

            if is_matched== False:
                sendemail.send_email(cur_img_path, "Security Alert - Door 2", "Unidentified Person")
            t2 = time.time()

            print(f'Time to find match: {t1 - t0}')
            print(f'Time to find match and email it: {t2 - t0}')

            # os.remove(cur_img_path)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
