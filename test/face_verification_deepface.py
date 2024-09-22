from deepface import DeepFace


class FaceVerification:
    def __init__(self):
        self.backends = [
          'opencv',
          'ssd',
          'dlib',
          'mtcnn',
          'fastmtcnn',
          'retinaface',
          'mediapipe',
          'yolov8',
          'yunet',
          'centerface',
        ]
        self.model = DeepFace

    def __call__(self, ref, gen):
        set1 = [i for i in ref[0].numpy().transpose((1, 2, 3, 0))]
        set2 = [i for i in gen[0].numpy().transpose((1, 2, 3, 0))]
        data = [(a, b) for a, b in zip(set1, set2)]

        results = [
                self.model.verify(
                        img1_path=pair[0],
                        img2_path=pair[1],
                        enforce_detection=False,
                        detector_backend='retinaface'
                ) for pair in data
        ]
        return results


if __name__ == '__main__':
    from dataset import CustomDataset

    deepface = FaceVerification()

    set1_dataset1 = CustomDataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test1\\set_1", backsub=False)
    set2_dataset1 = CustomDataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test1\\set_2", backsub=False)
    dist_value1 = deepface(set1_dataset1, set2_dataset1)
    print('Test 1:', dist_value1)

    set1_dataset2 = CustomDataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test2\\set_1", backsub=False)
    set2_dataset2 = CustomDataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test2\\set_2", backsub=False)
    dist_value2 = deepface(set1_dataset2, set2_dataset2)
    print('Test 2:', dist_value2)

    set1_dataset3 = CustomDataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test3\\set_1", backsub=False)
    set2_dataset3 = CustomDataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test3\\set_2", backsub=False)
    dist_value3 = deepface(set1_dataset3, set2_dataset3)
    print('Test 3:', dist_value3)

    # TODO DO SOME BAD EXAMPLES FIRST
    # cosine distance doesn't really work well here but i dunno why
