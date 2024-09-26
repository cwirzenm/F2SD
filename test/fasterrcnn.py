import cv2
import cvzone
import torch
import os
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn_v2, FasterRCNN_ResNet50_FPN_V2_Weights


class FasterRCNNResnet50V2:

    def __init__(self, cuda=True):
        self.fasterRCNN = fasterrcnn_resnet50_fpn_v2(weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
        self.fasterRCNN.eval()
        if cuda and torch.cuda.is_available(): self.fasterRCNN.to('cuda')

        self.classnames = []
        with open('../lib/classes.txt', 'r') as f: self.classnames = f.read().splitlines()

    def __call__(self, img_path):
        image = cv2.imread(img_path)
        image = cv2.resize(image, (640, 480))

        with torch.no_grad():
            tensor_img = transforms.ToTensor()(image).cuda()
            pred = self.fasterRCNN([tensor_img])

            bbox, scores, labels = pred[0]['boxes'], pred[0]['scores'], pred[0]['labels']
            conf = torch.argwhere(scores > 0.70).shape[0]
            for i in range(conf):
                x, y, w, h = bbox[i].cpu().numpy().astype('int')
                classname = labels[i].cpu().numpy().astype('int')
                class_detected = self.classnames[classname]
                cv2.rectangle(image, (x, y), (w, h), (0, 0, 255), 4)
                cvzone.putTextRect(image, class_detected, [x + 8, y - 12], scale=2, border=2)
                cv2.imwrite('data1.png', image)

        cv2.imshow('frame', image)
        cv2.waitKey(0)


if __name__ == '__main__':
    model = FasterRCNNResnet50V2()

    # stock images realistic
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_1/row-1-column-1.jpg'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_1/row-1-column-2.jpg'
    # model(ref_path)

    # # stock image vs Tom Hanks
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_2/row-1-column-5.jpg'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_2/gettyimages-1257937597.jpg'
    # print(model(ref_path, gen_path), end='\n\n')  # good threshold is 0.6
    #
    # # Geralt of Rivia
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test2/set_1/im_1.png'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test2/set_1/im_2.png'
    # print(model(ref_path, gen_path), end='\n\n')  # good threshold is 0.6

    # Flintstones note fail
    ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/temporalstory/gt_flintstones/row-3-column-2.png'
    gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/temporalstory/gt_flintstones/row-16-column-4.png'
    model(ref_path)
    model(gen_path)

    # base_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test4'
    # for f in os.listdir(base_path):
    #     model(os.path.join(base_path, f))
