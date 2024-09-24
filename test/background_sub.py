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
        self.mask = None

    def __call__(self, img, cuda=True):
        preprocess = Compose([
                ToDtype(torch.float32, scale=True),
                Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        """1. Generate a mask using DeepLabV3_ResNet50 model"""

        # create a mini-batch as expected by the model
        input_batch = preprocess(img).unsqueeze(0)

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            self.resnet.to('cuda')

        # get model output
        with torch.no_grad():
            output = self.resnet(input_batch)['out'][0]
        output_predictions = output.argmax(0)

        # create a binary mask based on the model output
        mask = output_predictions.byte().cpu().numpy()
        background = np.zeros(mask.shape)
        self.bin_mask = np.where(mask, 255, background).astype(np.uint8)

        """2. Subtract the generated mask from the input image to get the foreground"""

        # add an alpha channel to the image
        r, g, b = cv2.split(np.array(img).astype('uint8').transpose((1, 2, 0)))
        a = np.ones(self.bin_mask.shape, dtype='uint8') * 255
        alpha_img = cv2.merge([b, g, r, a], 4)

        # expand the mask to cover all channels
        new_mask = np.stack([self.bin_mask] * 4, axis=2)

        # apply the mask on the expanded image
        foreground = np.where(new_mask, alpha_img, np.zeros(alpha_img.shape)).astype(np.uint8)
        return foreground

    def get_mask(self): return self.bin_mask

    @staticmethod
    def read_image(file): return read_image(file, ImageReadMode.RGB)


if __name__ == '__main__':
    backsub = BackGroundSubtraction()
    img = backsub.read_image("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test4\\1.png")
    foreground = backsub(img)
    cv2.imshow('fg', foreground)
    cv2.waitKey(0)

    img = backsub.read_image("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test4\\2.png")
    foreground = backsub(img)
    cv2.imshow('fg', foreground)
    cv2.waitKey(0)

    img = backsub.read_image("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test4\\3.png")
    foreground = backsub(img)
    cv2.imshow('fg', foreground)
    cv2.waitKey(0)

    img = backsub.read_image("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test4\\4.png")
    foreground = backsub(img)
    cv2.imshow('fg', foreground)
    cv2.waitKey(0)

    img = backsub.read_image("C:\\Users\\mxnaz\\OneDrive\\Documents\\Bath Uni\\13 Dissertation\\data\\test4\\5.png")
    foreground = backsub(img)
    cv2.imshow('fg', foreground)
    cv2.waitKey(0)
