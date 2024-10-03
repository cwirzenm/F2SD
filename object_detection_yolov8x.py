import numpy as np
from ultralytics.utils.plotting import Annotator, colors
from ultralytics import YOLO
from torchvision.transforms.v2 import Compose, Normalize, ToDtype, Resize
import matplotlib.pyplot as plt
import torch
import cv2
import os


class YoloV8X:
    def __init__(self, return_cropped_frame=False):
        self.detection_model = YOLO('lib/yolov8x.pt')
        self.detection_model.to('cuda')

        self.embedding_model = YOLO('lib/yolov8x.pt')
        self.embedding_model.to('cuda')

        self.return_cropped_frame = return_cropped_frame

        # set model parameters
        self.detection_model.multi_label = False  # NMS multiple labels per box

    def __call__(self, frame_and_path: tuple, show=False):
        preprocess = Compose([
                ToDtype(torch.float32, scale=True),
                Resize((640, 640), antialias=True),
                # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        frame = frame_and_path[0]
        frame = preprocess(frame).unsqueeze(0)

        path = frame_and_path[1]

        # inference with test time augmentation
        result = self.detection_model(
                frame,
                imgsz=1280,
                augment=True,
                conf=0.6,
                iou=0.45,
                max_det=100,
                visualize=True,
                # classes= todo list of classes
        )[0]

        # show detection bounding boxes on image
        if show: result.show()

        boxes = result.boxes.data
        scores = result.boxes.conf
        categories = result.boxes.cls

        if boxes.shape[0] == 0:
            cropped_frame = frame
        else:
            areas = [(int(a[3]) - int(a[1])) * (int(a[2]) - int(a[0])) for a in boxes]

            # picking the object with the highest box area
            main_idx = int(areas.index(max(areas)))
            box = boxes[main_idx]

            # crop image and return
            cropped_frame = cv2.imread(path)[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
            # plt.imshow(cropped_frame[..., ::-1])

        if self.return_cropped_frame: return cropped_frame

        # cropped_frames = np.array(cropped_frames)

        # self.activations = {}
        # detection_head = self.model.model.model[-1]
        #
        # cv2_layer = detection_head.cv2[-1]
        # cv2_layer.register_forward_hook(self.get_forward_hook('cv2'))
        #
        # cv3_layer = detection_head.cv3[-1]
        # cv3_layer.register_forward_hook(self.get_forward_hook('cv3'))
        #
        # dfl_layer = detection_head.dfl
        # dfl_layer.register_forward_hook(self.get_forward_hook('dfl'))

        # rerun interference with a cropped image
        # inference with test time augmentation
        embeddings = self.embedding_model(
                cropped_frame,
                imgsz=256,
                agnostic_nms=True,
                retina_masks=True,
                embed=[21]
        )[0]
        embeddings = embeddings.cpu().numpy()
        return embeddings

    # def get_forward_hook(self, layer):
    #     def hook(model, input, output):
    #         layer_activations = []
    #         for i, activation in enumerate(output):
    #             layer_activations.append(activation.detach())
    #         self.activations[layer] = layer_activations
    #     return hook


if __name__ == '__main__':
    model = YoloV8X()

    # stock images realistic
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_1/row-1-column-1.jpg'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_1/row-1-column-2.jpg'
    # model(ref_path)    # model(gen_path)
    #
    # # stock image vs Tom Hanks
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_2/row-1-column-5.jpg'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_2/gettyimages-1257937597.jpg'
    # model(ref_path)
    # model(gen_path)

    # Geralt of Rivia
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test2/set_1/im_1.png'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test2/set_1/im_2.png'
    # x = model(ref_path)
    # y = model(gen_path)

    path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test2/set_1/'
    e = model(path)

    # # Flintstones
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/temporalstory/gt_flintstones/row-3-column-2.png'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/temporalstory/gt_flintstones/row-16-column-4.png'
    # model(ref_path)
    # model(gen_path)
    #
    # base_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test4'
    # for f in os.listdir(base_path):
    #     model(os.path.join(base_path, f))

    # base_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/tests'
    # data = model([os.path.join(base_path, f) for f in os.listdir(base_path)], get_activation=True)
    # for f in os.listdir(base_path):
    #     tensor = model(os.path.join(base_path, f))
    # break
