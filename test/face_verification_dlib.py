import dlib
import numpy as np
import cv2
import os


def getFace(img):
    face_detector = dlib.get_frontal_face_detector()
    return face_detector(img, 1)[0]


def encodeFace(image):
    face_location = getFace(image)
    pose_predictor = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
    face_landmarks = pose_predictor(image, face_location)
    face_encoder = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
    face = dlib.get_face_chip(image, face_landmarks)
    encodings = np.array(face_encoder.compute_face_descriptor(face))
    return encodings


def getSimilarity(image1, image2):
    face1_embeddings = encodeFace(image1)
    face2_embeddings = encodeFace(image2)
    return np.linalg.norm(face1_embeddings - face2_embeddings)


img1 = cv2.imread('C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_1/row-1-column-1.jpg')
img2 = cv2.imread('C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_1/row-1-column-2.jpg')

distance = getSimilarity(img1, img2)
print(distance)
if distance < .6:
    print("Faces are of the same person.")
else:
    print("Faces are of different people.")

# todo more testing!!!!!!!!!!!!!!!
# todo fuse with fsd maybe???
