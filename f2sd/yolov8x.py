from ultralytics import YOLO
from torchvision.transforms.v2 import Compose, ToDtype, Resize
import matplotlib.pyplot as plt
import torch
import cv2


class YoloV8X:
    def __init__(self):
        self.detection_model = YOLO('lib/yolov8x.pt')
        self.embedding_model = YOLO('lib/yolov8x.pt')

        # set model parameters
        self.detection_model.multi_label = False  # NMS multiple labels per box

    def __call__(self, frames: torch.Tensor, show=False) -> torch.Tensor:
        preprocess = Compose([
                ToDtype(torch.float32, scale=True),
                Resize((640, 640), antialias=True),
        ])

        frames = preprocess(frames).flatten(start_dim=0, end_dim=1)

        # inference with test time augmentation
        results = self.detection_model(
                frames,
                imgsz=640,
                augment=True,
                conf=0.6,
                iou=0.45,
                max_det=100,
                verbose=False
        )

        cropped_frames = []
        for idx, result in enumerate(results):
            if show: result.show()

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
                cropped_frames.append(cropped_frame)

                if show:
                    # convert from BGR to RGB for the image to be correctly displayed using plt
                    plt.imshow(cv2.cvtColor(cropped_frame.permute(1, 2, 0).numpy(), cv2.COLOR_BGR2RGB))

        resize_func = Resize((256, 256))
        cropped_frames = [resize_func(f) for f in cropped_frames]
        cropped_frames = torch.stack(cropped_frames)

        # rerun interference with a cropped image
        # inference with test time augmentation
        embeddings = self.embedding_model(
                cropped_frames,
                imgsz=256,
                agnostic_nms=True,
                retina_masks=True,
                verbose=False,
                embed=[21]
        )
        # target shape: (n, dims, 1, 1)
        embeddings = torch.stack(embeddings).unsqueeze(-1).unsqueeze(-1)
        return embeddings

    def eval(self):
        # need to manually set the models to `training` = False due to yolo's `eval()` being broken and causing training to start
        self.detection_model.training = False
        for module in self.detection_model.children():
            module.training = False

        self.embedding_model.training = False
        for module in self.embedding_model.children():
            module.training = False

    def to(self, device):
        self.detection_model.to(device)
        self.embedding_model.to(device)
