import numpy as np
from ultralytics.utils.plotting import Annotator, colors
from ultralytics import YOLO
import matplotlib.pyplot as plt
import torch
import cv2
import os


class YoloV8X:
    def __init__(self):
        self.detection_model = YOLO('lib/yolov8x.pt')
        self.detection_model.to('cuda')

        self.embedding_model = YOLO('lib/yolov8x.pt')
        self.embedding_model.to('cuda')

        # set model parameters
        self.detection_model.multi_label = False  # NMS multiple labels per box

    def __call__(self, frames, show=True):
        # inference with test time augmentation
        results = self.detection_model(
                frames,
                imgsz=1280,
                augment=True,
                conf=0.6,
                iou=0.45,
                max_det=100,
                visualize=True,
                # classes= todo list of classes
        )

        # show detection bounding boxes on image
        if show:
            for result in results:
                result.show()

        object_data = []
        cropped_frames = []
        for result in results:
            boxes = result.boxes.data
            scores = result.boxes.conf
            categories = result.boxes.cls
            areas = [(int(a[3]) - int(a[1])) * (int(a[2]) - int(a[0])) for a in boxes]

            object_data.append((boxes, scores, categories, areas))

            # picking the object with the highest box area
            main_idx = int(areas.index(max(areas)))
            box = boxes[main_idx]

            # crop image and return
            crop_obj = cv2.imread(result.path)[int(box[1]): int(box[3]), int(box[0]): int(box[2])]
            cropped_frames.append(crop_obj)
            # plt.imshow(crop_obj[..., ::-1])

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
                cropped_frames,
                imgsz=256,
                embed=[247, 266, 268]
        )
        embeddings = [embeddings[0].cpu().numpy().flatten()] + [e.cpu().numpy().flatten() for e in embeddings[1]]
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
    ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test2/set_1/im_1.png'
    gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test2/set_1/im_2.png'
    x = model(ref_path)
    y = model(gen_path)

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
