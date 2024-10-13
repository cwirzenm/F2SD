import os
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import Resize, Compose, ToDtype, Normalize
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, img_dir: str, res=(256, 256), mode='R2+1D', verbose=False, **kwargs):
        self.img_dir: str = img_dir
        self.res: tuple = res
        self.mode: str = mode
        self.verbose: bool = verbose
        self.frames: list = [[os.path.join(r, x) for x in f] for r, d, f in os.walk(img_dir) if f]

    def __str__(self) -> str: return str(self.frames)

    def __len__(self) -> int: return len(self.frames)

    def __getitem__(self, idx) -> torch.Tensor:
        # get the frame paths
        frame_paths: list[str] = [p for p in self.frames[idx]]

        # get the frames
        if self.mode == 'R2+1D':
            frames: list[torch.Tensor] = [read_image(i, ImageReadMode.RGB) for i in frame_paths]
        elif self.mode == 'YOLO':
            frames = []
            for i in frame_paths:
                r, g, b = read_image(i, ImageReadMode.RGB).split(1, 0)
                img = torch.stack((b, g, r)).squeeze()
                frames.append(img)
        else: raise NotImplementedError

        # transform the frames
        vid: list[torch.Tensor] = [self.transform()(f) for f in frames]  # each frame must be (C, H, W)

        # double-check the shape
        assert vid[0].shape[0] == 3, f"Wrong shape {vid[0].shape}. The shape ought to be (3, 64, 64)"

        if self.mode == 'R2+1D':
            # move channels to the first axis
            vid: torch.Tensor = torch.stack(vid).permute(1, 0, 2, 3)  # (C, T, H, W)
        elif self.mode == 'YOLO':
            vid: torch.Tensor = torch.stack(vid)  # do not permute for YOLO
        else:
            raise NotImplementedError

        return vid

    def print_shape(self, img) -> torch.Tensor:
        if self.verbose: print(f"{img.shape} {type(img)}")
        return img

    def transform(self) -> Compose:
        stack = [
            ToDtype(torch.float32, scale=True),
            Resize(self.res, antialias=True),
            lambda x: x[:3, ::],
        ]
        if self.mode == 'R2+1D':
            # standard normalisation for R(2+1)D model
            stack.append(Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]))

        return Compose(stack)
