import dlib
import numpy as np
import cv2
import os
import itertools


class FaceVerification:
    def __init__(self):
        self.pose_predictor_model = 'lib/shape_predictor_5_face_landmarks.dat'
        self.face_recognition_model = 'lib/dlib_face_recognition_resnet_model_v1.dat'
        self.face_detector_model = 'lib/mmod_human_face_detector.dat'
        self.face_detector = dlib.get_frontal_face_detector()

    def __call__(self, ref, gen):
        img1 = cv2.imread(ref)
        img2 = cv2.imread(gen)

        face1_embeddings = self._encodeFace(img1)
        face2_embeddings = self._encodeFace(img2)
        if face1_embeddings is False or face2_embeddings is False: return 2.0
        return np.linalg.norm(face1_embeddings - face2_embeddings)

    def _encodeFace(self, image):
        pose_predictor = dlib.shape_predictor(self.pose_predictor_model)
        face_encoder = dlib.face_recognition_model_v1(self.face_recognition_model)
        face_detector = dlib.cnn_face_detection_model_v1(self.face_detector_model)
        face_location = self._getFace(face_detector, image)
        if not face_location: return False
        face_landmarks = pose_predictor(image, face_location)
        face = dlib.get_face_chip(image, face_landmarks)
        encodings = np.array(face_encoder.compute_face_descriptor(face))
        return encodings

    def _getFace(self, face_detector, image):
        try:
            return self.face_detector(image, 1)[0]
            # return face_detector(image, 1)[0].rect
        except IndexError:
            print('Face not detected.')
            return False


if __name__ == '__main__':
    model = FaceVerification()

    # stock images realistic
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_1/row-1-column-1.jpg'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_1/row-1-column-2.jpg'
    # print(model(ref_path, gen_path), end='\n\n')  # good threshold is 0.6
    #
    # # stock image vs Tom Hanks
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_2/row-1-column-5.jpg'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_2/gettyimages-1257937597.jpg'
    # print(model(ref_path, gen_path), end='\n\n')  # good threshold is 0.6
    #
    # # Geralt of Rivia
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test2/set_1/im_1.png'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test2/set_1/im_2.png'
    # print(model(ref_path, gen_path), end='\n\n')  # good threshold is 0.6
    #
    # # Flintstones note fail
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/temporalstory/gt_flintstones/row-3-column-2.png'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/temporalstory/gt_flintstones/row-16-column-4.png'
    # print(model(ref_path, gen_path), end='\n\n')  # good threshold is 0.6

    base_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test4'
    dir = os.listdir(base_path)
    for f1, f2 in itertools.combinations(dir, 2):
        print(f"Comparing {f1} and {f2}.")
        print(model(os.path.join(base_path, f1), os.path.join(base_path, f2)), end='\n\n')
