import os
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import Resize, Compose, ToDtype, Normalize
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, img_dir, res=(64, 64), verbose=False):
        self.img_dir = img_dir
        self.res = res
        self.verbose = verbose
        self.frames = [[os.path.join(r, x) for x in f] for r, d, f in os.walk(img_dir) if f]

    def __str__(self): return str(self.frames)

    def __len__(self): return len(self.frames)

    def __getitem__(self, idx):
        # get the frame paths
        frame_paths = [p for p in self.frames[idx]]

        # get the frames
        frames = [read_image(i, ImageReadMode.RGB) for i in frame_paths]

        # transform the frames
        vid = [self.transform()(f) for f in frames]  # each frame must be (C, H, W)

        # double-check the shape
        assert vid[0].shape[0] == 3, f"Wrong shape {vid[0].shape}. The shape ought to be (3, 64, 64)"

        # stack and move channels to the first axis
        vid = torch.stack(vid).permute(1, 0, 2, 3)  # (C, T, H, W)
        return vid

    def print_shape(self, img):
        if self.verbose: print(f"{img.shape} {type(img)}")
        return img

    def transform(self) -> Compose:
        return Compose([
                self.print_shape,

                # models like floats
                ToDtype(torch.float32, scale=True),

                # R(2+1)D likes 64x64 inputs
                Resize(self.res, antialias=True),
                self.print_shape,

                # trim excessive channels
                lambda x: x[:3, ::],

                # standard normalisation for R(2+1)D model
                Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])
