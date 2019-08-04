import cv2
import time
import sys
import os
import numpy as np
import urllib
from PIL import Image
import tensorflow.contrib.tensorrt as trt
import glob
import face_recognition
import send_email_attach_aws_ses as sendemail

# Our list of known face encodings
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

def save_image(frame):
    timestamp = int(time.time())
    impath = "images-visitors/face-" + str(timestamp) + ".jpg"

    cv2.imwrite(impath, frame)
    return impath

def check_if_matched(score, threshold=0.5):
    print(f'Best match score {score}')
    if score <= threshold:
        return True
    else:
        return False

# Load face detection model
cap = cv2.VideoCapture(1)
face_cascade = cv2.CascadeClassifier('/opt/opencv/data/haarcascades/haarcascade_frontalface_default.xml')
print("Loaded cascade classifier model")

# load encodings for all the known faces into memory
load_known_faces()
print("Loaded known faces encodings")

# create a folder to store all visitor images
if not os.path.exists('images-visitors'):
    os.mkdir('images-visitors')

# Continuously capture images from the camera at regular interval (every 2 seconds)
# If a face is detected in the the image,
# and check if it matches with any of the known faces.
# If it's an unkown face or a person, send an alert email with the person's picture.
while(True):
    ret, frame = cap.read()

    time.sleep(2)
    if ret == True:
        print("captured image")
        gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Face detection. Check if there's a face in the image.
        faces = face_cascade.detectMultiScale(gray_img, 1.3, 5)
        for (x, y, w, h) in faces:
            t0 = time.time()

            # save image
            cur_img_path = save_image(frame)
            # resize image for faster retreival of encodings
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            face_locations = face_recognition.face_locations(small_frame)
            face_encodings = face_recognition.face_encodings(small_frame, face_locations)

            # get the consine similarity for the best match among all the known faces
            score = lookup_known_face(face_encodings[0])

            # check if the score is less than threshold i.e.,if the person is a
            # known person
            is_matched = check_if_matched(score)

            t1 = time.time()
            print(f'Time to find match: {t1 - t0}')

            if is_matched== False:
                print("Unknown person at Door 2")
                sendemail.send_email(cur_img_path, "Security Alert - Door 2", "Unidentified Person")
            t2 = time.time()
            print(f'Time to find match and email it: {t2 - t0}')

            # os.remove(cur_img_path)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
