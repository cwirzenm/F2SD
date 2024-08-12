import os
import torch
import functools
from PIL import Image
from torchvision.io import read_image, ImageReadMode
from torchvision import transforms
from torch.utils.data import Dataset


class CustomDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.frames = [[f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]]
        self.transform = transform if transform else self.default_transform()

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame_paths = [os.path.join(self.img_dir, p) for p in self.frames[idx]]
        frames = [read_image(i, ImageReadMode.RGB).numpy() for i in frame_paths]
        vid = [self.transform(f) for f in frames]
        vid = torch.stack(vid).permute(1, 0, 2, 3)
        return vid

    @staticmethod
    def default_transform() -> transforms.Compose:
        return transforms.Compose([
                functools.partial(Image.fromarray, mode='RGB'),
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                lambda x: x[:3, ::],
                # standard normalisation for R(2+1)D model
                transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
        ])
