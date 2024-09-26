import torch
import os


class YoloV5:
    def __init__(self):
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        self.model.to('cuda')

        print([x for x in self.model.named_modules()])
        print(self.model)

        # set model parameters
        self.model.conf = 0.25  # NMS confidence threshold
        self.model.iou = 0.45  # NMS IoU threshold
        self.model.agnostic = False  # NMS class-agnostic
        self.model.multi_label = False  # NMS multiple labels per box
        self.model.max_det = 3  # maximum number of detections per image

        self.activation = {}

        self.model.model.model.model[24].register_forward_hook(self.get_activation('model.model.model.24.m.2'))

    def __call__(self, img):
        # perform inference
        results = self.model(img)

        # inference with larger input size
        results = self.model(img, size=1280)

        # inference with test time augmentation
        results = self.model(img, augment=True)

        # parse results
        predictions = results.pred[0]
        boxes = predictions[:, :4]  # x1, y1, x2, y2
        scores = predictions[:, 4]
        categories = predictions[:, 5]

        # show detection bounding boxes on image
        results.show()

        # save results into "results/" folder
        # results.save(save_dir='results/')

        return self.activation

    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output[0].detach()

        return hook


if __name__ == '__main__':
    model = YoloV5()

    # stock images realistic
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_1/row-1-column-1.jpg'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_1/row-1-column-2.jpg'
    # model(ref_path)    # model(gen_path)
    #
    # # stock image vs Tom Hanks
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_2/row-1-column-5.jpg'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test3/set_2/gettyimages-1257937597.jpg'
    # model(ref_path)
    # model(gen_path)
    #
    # # Geralt of Rivia
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test2/set_1/im_1.png'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test2/set_1/im_2.png'
    # model(ref_path)
    # model(gen_path)
    #
    # # Flintstones
    # ref_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/temporalstory/gt_flintstones/row-3-column-2.png'
    # gen_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/temporalstory/gt_flintstones/row-16-column-4.png'
    # model(ref_path)
    # model(gen_path)
    #
    # base_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/test4'
    # for f in os.listdir(base_path):
    #     model(os.path.join(base_path, f))

    base_path = 'C:/Users/mxnaz/OneDrive/Documents/Bath Uni/13 Dissertation/data/tests'
    for f in os.listdir(base_path):
        tensor = model(os.path.join(base_path, f))
        break
