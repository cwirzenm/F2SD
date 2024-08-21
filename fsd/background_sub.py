import cv2
import numpy as np
import torch
from torchvision.io import read_image, ImageReadMode
from torchvision.transforms.v2 import Compose, Normalize, ToDtype
from torchvision.models.segmentation import deeplabv3_resnet50, DeepLabV3_ResNet50_Weights


class BackGroundSubtraction:

    def __init__(self):
        self.resnet = deeplabv3_resnet50(weights=DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
        self.resnet.eval()

    def make_transparent_foreground(self, pic, mask):
        # split the image into channels
        b, g, r = cv2.split(np.array(pic).astype('uint8').transpose((1, 2, 0)))
        # add an alpha channel with and fill all with transparent pixels (max 255)
        a = np.ones(mask.shape, dtype='uint8') * 255
        # merge the alpha channel back
        alpha_im = cv2.merge([b, g, r, a], 4)
        # create a transparent background
        bg = np.zeros(alpha_im.shape)
        # setup the new mask
        new_mask = np.stack([mask, mask, mask, mask], axis=2)
        # copy only the foreground color pixels from the original image where mask is set
        foreground = np.where(new_mask, alpha_im, bg).astype(np.uint8)

        return foreground

    def remove_background(self, input_file):
        input_image = read_image(input_file, ImageReadMode.RGB)
        # input_image = Image.open(input_file)
        preprocess = Compose([
                # ToTensor(),
                ToDtype(torch.float32, scale=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        input_tensor = preprocess(input_image)
        input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.resnet.to('cuda')

        with torch.no_grad():
            output = self.resnet(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # create a binary (black and white) mask of the profile foreground
        mask = output_predictions.byte().cpu().numpy()
        background = np.zeros(mask.shape)
        bin_mask = np.where(mask, 255, background).astype(np.uint8)

        foreground = self.make_transparent_foreground(input_image, bin_mask)

        return foreground, bin_mask


if __name__ == '__main__':
    x = BackGroundSubtraction()
    foreground, bin_mask = x.remove_background("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test2\\set_2\\im_2.png")
    cv2.imshow('fg', foreground)
    cv2.imshow('bin', bin_mask)
    cv2.waitKey(0)


