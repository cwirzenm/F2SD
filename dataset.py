import os
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import Resize, Compose, ToImage
from torch.utils.data import Dataset
from test.background_sub import BackGroundSubtraction


class CustomDataset(Dataset):
    def __init__(self, img_dir, backsub=False, verbose=False):
        self.img_dir = img_dir
        self.frames = [[f for f in os.listdir(img_dir) if os.path.isfile(os.path.join(img_dir, f))]]
        self.backsub = BackGroundSubtraction() if backsub else lambda x: x
        self.verbose = verbose

    def __len__(self): return len(self.frames)

    def __getitem__(self, idx):
        # get the frame paths
        frame_paths = [os.path.join(self.img_dir, p) for p in self.frames[idx]]

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
        if self.verbose: print(img.shape)
        return img

    def transform(self) -> Compose:
        return Compose([
                self.print_shape,

                # apply background subtraction if applicable
                self.backsub,
                self.print_shape,

                # convert to tensor (really only needed for backsub)
                ToImage(),
                self.print_shape,

                # RandomRotation((20, 40)),

                # models like floats
                # ToDtype(torch.float32, scale=True),

                # R(2+1)D likes 64x64 inputs
                # Resize((64, 64), antialias=True),
                Resize((400, 400), antialias=True),
                self.print_shape,

                # trim excessive channels
                lambda x: x[:3, ::],

                # standard normalisation for R(2+1)D model
                # Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])


if __name__ == "__main__":
    test1_dataset = CustomDataset(
            "C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test2\\set_2",
            backsub=False,
            verbose=True
    )
    a = [b for b in test1_dataset]
