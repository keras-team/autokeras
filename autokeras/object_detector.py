# ----------------------------------
# The idea of the classes and functions in this file is largely borrowed from
# https://github.com/amdegroot/ssd.pytorch
# A huge thank you to the authors: Max deGroot and Ellis Brown
# ----------------------------------

from __future__ import print_function
from autokeras.pretrained import Pretrained

from autokeras.object_detection.data import *
from autokeras.object_detection.data import VOC_CLASSES
from autokeras.object_detection.data import base_transform
from autokeras.object_detection.ssd import multibox, vgg, add_extras, SSD
from autokeras.utils import download_file, temp_path_generator, get_device
from autokeras.constant import Constant
import os
import torch
from torch.autograd import Variable
import numpy as np
import cv2
from matplotlib import pyplot as plt


class ObjectDetector(Pretrained):
    def __init__(self):
        super(ObjectDetector, self).__init__()
        self.model = None
        self.device = get_device()

    def _build_ssd(self, phase, size=300, num_classes=21):
        if phase != "test" and phase != "train":
            print("ERROR: Phase: " + phase + " not recognized")
            return
        if size != 300:
            print("ERROR: You specified size " + repr(size) + ". However, " +
                  "currently only SSD300 (size=300) is supported!")
            return
        base = {
            '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                    512, 512, 512],
            '512': [],
        }
        extras = {
            '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
            '512': [],
        }
        mbox = {
            '300': [4, 6, 6, 6, 4, 4],  # number of boxes per feature map location
            '512': [],
        }

        base_, extras_, head_ = multibox(vgg(base[str(size)], 3),
                                         add_extras(extras[str(size)], 1024),
                                         mbox[str(size)], num_classes)
        return SSD(phase, size, base_, extras_, head_, num_classes, self.device)

    def load(self, model_path=None):
        # https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
        if model_path is None:
            file_link = Constant.PRE_TRAIN_DETECTION_FILE_LINK
            # model_path = os.path.join(temp_path_generator(), "object_detection_pretrained.pth")
            model_path = temp_path_generator() + '_object_detection_pretrained.pth'
            download_file(file_link, model_path)
        # load net
        num_classes = len(VOC_CLASSES) + 1  # +1 for background
        self.model = self._build_ssd('test', 300, num_classes)  # initialize SSD
        if self.device.startswith("cuda"):
            self.model.load_state_dict(torch.load(model_path))
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        self.model.eval()
        print('Finished loading model!')

        self.model = self.model.to(self.device)

    def predict(self, img_path, output_file_path=None):
        """
        
        Returns:
            List of tuples. Each tuple is like ((x1, y1), (h, w), category, confidence).
        """
        from matplotlib.ticker import NullLocator

        dataset_mean = (104, 117, 123)

        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = base_transform(rgb_image, 300, dataset_mean)
        x = x.astype(np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = Variable(x.unsqueeze(0))  # wrap tensor in Variable
        # if self.device.startswith("cuda"):
        xx = xx.to(self.device)
        y = self.model(xx)

        # (batch, num_classes, top_k, 5), 5 means (confidence, )
        detections = y.data
        results = []
        # scale each detection back up to the image
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        for i in range(detections.size(1)):
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                score = detections[0, i, j, 0].item()
                label_name = VOC_CLASSES[i - 1]
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                result = ((pt[0], pt[1]), (pt[2] - pt[0] + 1, pt[3] - pt[1] + 1), label_name, score)
                results.append(set(result))
                j += 1

        if output_file_path is not None:
            # plt.figure(figsize=(10,10))
            colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
            plt.imshow(rgb_image)  # plot the image for matplotlib
            current_axis = plt.gca()
            current_axis.set_axis_off()
            current_axis.xaxis.set_major_locator(NullLocator())
            current_axis.yaxis.set_major_locator(NullLocator())

            # scale each detection back up to the image
            for i in range(detections.size(1)):
                j = 0
                while detections[0, i, j, 0] >= 0.6:
                    score = detections[0, i, j, 0]
                    label_name = VOC_CLASSES[i - 1]
                    display_txt = '%s: %.2f' % (label_name, score)
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    coords = (pt[0], pt[1]), pt[2] - pt[0] + 1, pt[3] - pt[1] + 1
                    color = colors[i]
                    current_axis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                    current_axis.text(pt[0], pt[1], display_txt, bbox={'facecolor': color, 'alpha': 0.5})
                    j += 1
            plt.axis('off')
            plt.tight_layout()
            save_name = img_path.split('/')[-1]
            save_name = save_name.split('.')
            save_name = '.'.join(save_name[:-1]) + "_prediction." + save_name[-1]
            plt.savefig(os.path.join(output_file_path, save_name), bbox_inches='tight', pad_inches=0)
            plt.clf()

        return results
