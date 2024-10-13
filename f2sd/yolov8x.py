import numpy as np
from ultralytics.utils.plotting import Annotator, colors
from ultralytics import YOLO
from torchvision.transforms.v2 import Compose, Normalize, ToDtype, Resize, ToImage
from torchvision.io import read_image, ImageReadMode
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

    def __call__(self, frames: torch.Tensor, show=False) -> torch.Tensor:
        preprocess = Compose([
                ToDtype(torch.float32, scale=True),
                Resize((640, 640), antialias=True),
        ])

        frames = preprocess(frames)
        # frames = preprocess(frames).unsqueeze(0)

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

        cropped_frames = []
        for idx, result in enumerate(results):
            if show: result.show()

            # todo why is it grayscale?
            boxes = result.boxes.data
            scores = result.boxes.conf
            categories = result.boxes.cls

            if boxes.shape[0] == 0: cropped_frames.append(frames[idx])
            else:
                areas = [(int(a[3]) - int(a[1])) * (int(a[2]) - int(a[0])) for a in boxes]

                # picking the object with the highest box area
                main_idx = int(areas.index(max(areas)))
                box = boxes[main_idx]

                # crop image and return
                cropped_frame = frames[idx][:, int(box[1]): int(box[3]), int(box[0]): int(box[2])]
                if show:
                    # convert from BGR to RGB for the image to be correctly displayed using plt
                    plt.imshow(cv2.cvtColor(cropped_frame.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB))

        cropped_frames = torch.stack(cropped_frames)

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
                agnostic_nms=True,
                retina_masks=True,
                embed=[21]
        )
        # todo to unsqueeze or not to unsqueeze????
        embeddings = embeddings[0].unsqueeze(0).cpu().numpy()
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

    consistory_2_path = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\consistory\\1\\target"
    consistory_2 = []
    for i in [os.path.join(consistory_2_path, f) for f in os.listdir(consistory_2_path)]:
        img = read_image(i, ImageReadMode.RGB)
        r, g, b = img.split(1, 0)
        img = torch.stack((b, g, r)).squeeze()
        consistory_2.append(img)
    consistory_2 = torch.stack(consistory_2)
    embeddings_2 = model(consistory_2)

    print(embeddings_2)

    consistory_1_path = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\consistory\\1\\source"
    consistory_1 = []
    for i in [os.path.join(consistory_1_path, f) for f in os.listdir(consistory_1_path)]:
        img = read_image(i, ImageReadMode.RGB)
        r, g, b = img.split(1, 0)
        img = torch.stack((b, g, r)).squeeze()
        consistory_1.append(img)
    consistory_1 = torch.stack(consistory_1)
    embeddings_1 = model(consistory_1, show=True)

    # storydalle_flintstones = "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\f2sd_data\\storydalle\\flintstones"
