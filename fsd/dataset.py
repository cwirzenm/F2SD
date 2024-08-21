import os
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import ToDtype, Resize, Normalize, Compose
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
        frames = [read_image(i, ImageReadMode.RGB) for i in frame_paths]  # (T, C, H, W)

        # frames = [cv2.imread(i) for i in frame_paths]  # (T, H, W, C)
        # backsub_knn = cv2.createBackgroundSubtractorMOG2()
        # vid = backsub_knn.apply(frames[0])
        # cv2.imshow('backsub_knn', vid)
        # cv2.waitKey(0)

        vid = [self.transform(f) for f in frames]
        # shape here should be (C, H, W)
        vid = torch.stack(vid).permute(1, 0, 2, 3)
        # output shape (C, T, H, W)
        return vid

    @staticmethod
    def default_transform() -> Compose:
        def print_shape(x):
            # print(x.shape)
            return x

        """Stack of transformations for dataset preprocessing"""
        return Compose([
                print_shape,
                ToDtype(torch.float32, scale=True),
                Resize((112, 112), antialias=True),
                print_shape,
                lambda x: x[:3, ::],
                print_shape,
                # standard normalisation for R(2+1)D model
                Normalize(mean=[0.43216, 0.394666, 0.37645], std=[0.22803, 0.22145, 0.216989]),
        ])


if __name__ == "__main__":
    test1_dataset = CustomDataset("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test2\\set_2")
    a = [b for b in test1_dataset]
