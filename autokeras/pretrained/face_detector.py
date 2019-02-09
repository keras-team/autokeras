# This is DFace's implementation of MTCNN modified for AutoKeras
# Link to DFace: https://github.com/kuaikuaikim/DFace
import os

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.autograd.variable import Variable

from autokeras.constant import Constant
from autokeras.pretrained.base import Pretrained
from autokeras.utils import get_device, download_file_from_google_drive, temp_path_generator, ensure_dir


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight.data)
        nn.init.constant_(m.bias, 0.1)


class PNet(nn.Module):

    def __init__(self):
        super(PNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(10, 16, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=1),
            nn.PReLU()
        )
        self.conv4_1 = nn.Conv2d(32, 1, kernel_size=1, stride=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1, stride=1)
        self.conv4_3 = nn.Conv2d(32, 10, kernel_size=1, stride=1)

        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        label = torch.sigmoid(self.conv4_1(x))
        offset = self.conv4_2(x)
        return label, offset


class RNet(nn.Module):

    def __init__(self):
        super(RNet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 28, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(28, 48, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 64, kernel_size=2, stride=1),
            nn.PReLU()

        )
        self.conv4 = nn.Linear(64 * 2 * 2, 128)
        self.prelu4 = nn.PReLU()
        self.conv5_1 = nn.Linear(128, 1)
        self.conv5_2 = nn.Linear(128, 4)
        self.conv5_3 = nn.Linear(128, 10)
        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv4(x)
        x = self.prelu4(x)
        det = torch.sigmoid(self.conv5_1(x))
        box = self.conv5_2(x)
        return det, box


class ONet(nn.Module):

    def __init__(self):
        super(ONet, self).__init__()

        self.pre_layer = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.PReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=2, stride=1),
            nn.PReLU()
        )
        self.conv5 = nn.Linear(128 * 2 * 2, 256)
        self.prelu5 = nn.PReLU()
        self.conv6_1 = nn.Linear(256, 1)
        self.conv6_2 = nn.Linear(256, 4)
        self.conv6_3 = nn.Linear(256, 10)
        self.apply(weights_init)

    def forward(self, x):
        x = self.pre_layer(x)
        x = x.view(x.size(0), -1)
        x = self.conv5(x)
        x = self.prelu5(x)
        det = torch.sigmoid(self.conv6_1(x))
        box = self.conv6_2(x)
        landmark = self.conv6_3(x)
        return det, box, landmark


def get_square_bbox(bbox):
    square_bbox = bbox.copy()

    h = bbox[:, 3] - bbox[:, 1] + 1
    w = bbox[:, 2] - bbox[:, 0] + 1
    l = np.maximum(h, w)
    square_bbox[:, 0] = bbox[:, 0] + w * 0.5 - l * 0.5
    square_bbox[:, 1] = bbox[:, 1] + h * 0.5 - l * 0.5

    square_bbox[:, 2] = square_bbox[:, 0] + l - 1
    square_bbox[:, 3] = square_bbox[:, 1] + l - 1
    return square_bbox


def generate_bounding_box(map, reg, scale, threshold):
    stride = 2
    cellsize = 12

    t_index = np.where(map > threshold)

    if t_index[0].size == 0:
        return np.array([])

    dx1, dy1, dx2, dy2 = [reg[0, t_index[0], t_index[1], i] for i in range(4)]
    reg = np.array([dx1, dy1, dx2, dy2])

    score = map[t_index[0], t_index[1], 0]
    boundingbox = np.vstack([np.round((stride * t_index[1]) / scale),
                             np.round((stride * t_index[0]) / scale),
                             np.round((stride * t_index[1] + cellsize) / scale),
                             np.round((stride * t_index[0] + cellsize) / scale),
                             score,
                             reg
                             ])

    return boundingbox.T


def resize_image(img, scale):
    height, width, channels = img.shape
    new_height = int(height * scale)
    new_width = int(width * scale)
    new_dim = (new_width, new_height)
    img_resized = cv2.resize(img, new_dim, interpolation=cv2.INTER_LINEAR)
    return img_resized


def pad(bboxes, w, h):
    tmpw = (bboxes[:, 2] - bboxes[:, 0] + 1).astype(np.int32)
    tmph = (bboxes[:, 3] - bboxes[:, 1] + 1).astype(np.int32)
    numbox = bboxes.shape[0]

    dx = np.zeros((numbox,))
    dy = np.zeros((numbox,))
    edx, edy = tmpw.copy() - 1, tmph.copy() - 1

    x, y, ex, ey = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]

    tmp_index = np.where(ex > w - 1)
    edx[tmp_index] = tmpw[tmp_index] + w - 2 - ex[tmp_index]
    ex[tmp_index] = w - 1

    tmp_index = np.where(ey > h - 1)
    edy[tmp_index] = tmph[tmp_index] + h - 2 - ey[tmp_index]
    ey[tmp_index] = h - 1

    tmp_index = np.where(x < 0)
    dx[tmp_index] = 0 - x[tmp_index]
    x[tmp_index] = 0

    tmp_index = np.where(y < 0)
    dy[tmp_index] = 0 - y[tmp_index]
    y[tmp_index] = 0

    return_list = [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph]
    return_list = [item.astype(np.int32) for item in return_list]

    return return_list


def nms(dets, thresh, mode="Union"):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        if mode == "Union":
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
        elif mode == "Minimum":
            ovr = inter / np.minimum(areas[i], areas[order[1:]])

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def convert_image_to_tensor(image):
    transform = transforms.ToTensor()
    return transform(image)


def convert_chw_tensor_to_hwc_numpy(tensor):
    if isinstance(tensor, Variable):
        return np.transpose(tensor.data.numpy(), (0, 2, 3, 1))
    elif isinstance(tensor, torch.FloatTensor):
        return np.transpose(tensor.numpy(), (0, 2, 3, 1))
    else:
        raise Exception("covert b*c*h*w tensor to b*h*w*c numpy error.This tensor must have 4 dimension.")


def vis_face(im_array, dets, output_file_path, landmarks=None):
    fig, ax = plt.subplots(1)
    ax.imshow(im_array)

    for i in range(dets.shape[0]):
        bbox = dets[i, :4]

        rect = plt.Rectangle((bbox[0], bbox[1]),
                             bbox[2] - bbox[0],
                             bbox[3] - bbox[1], fill=False,
                             edgecolor='yellow', linewidth=0.9)
        ax.add_patch(rect)

    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            landmarks_one = landmarks[i, :]
            landmarks_one = landmarks_one.reshape((5, 2))
            for j in range(5):
                cir1 = patches.Circle(xy=(landmarks_one[j, 0], landmarks_one[j, 1]), radius=2, alpha=0.4, color="red")
                ax.add_patch(cir1)
        plt.axis('off')
        fig.savefig(output_file_path, bbox_inches='tight', pad_inches=0)


class FaceDetector(Pretrained):
    """A class to predict faces using the MTCNN pre-trained model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pnet, rnet, onet = (torch.load(path, map_location=lambda storage, loc: storage) for path in self.local_paths)

        self.pnet_detector = PNet()
        self.pnet_detector.load_state_dict(pnet)
        self.pnet_detector = self.pnet_detector.to(self.device)
        self.pnet_detector.eval()

        self.rnet_detector = RNet()
        self.rnet_detector.load_state_dict(rnet)
        self.rnet_detector = self.rnet_detector.to(self.device)
        self.rnet_detector.eval()

        self.onet_detector = ONet()
        self.onet_detector.load_state_dict(onet)
        self.onet_detector = self.onet_detector.to(self.device)
        self.onet_detector.eval()

        self.min_face_size = 24
        self.stride = 2
        self.threshold = [0.6, 0.7, 0.7]
        self.scale_factor = 0.709

    @property
    def _google_drive_files(self):
        return Constant.FACE_DETECTOR_MODELS

    def predict(self, img_path, output_file_path=None):
        """Predicts faces in an image.

        Args:
            img_path: A string. The path to the image on which the prediction is to be done.
            output_file_path: A string. The path where the output image is to be saved after the prediction. `None` by default.

        Returns:
            A tuple containing numpy arrays of bounding boxes and landmarks. Bounding boxes are of shape `(n, 5)` and
            landmarks are of shape `(n, 10)` where `n` is the number of faces predicted. Each bounding box is of length
            5 and the corresponding rectangle is defined by the first four values. Each bounding box has five landmarks
            represented by 10 coordinates.
        """
        if not os.path.exists(img_path):
            raise ValueError('Image does not exist')
        img = cv2.imread(img_path)
        img_bg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        bounding_boxes, landmarks = self.detect_face(img)
        if output_file_path is not None:
            vis_face(img_bg, bounding_boxes, output_file_path, landmarks)
        return bounding_boxes, landmarks

    def detect_pnet(self, im):
        h, w, c = im.shape
        net_size = 12

        current_scale = float(net_size) / self.min_face_size
        im_resized = resize_image(im, current_scale)
        current_height, current_width, _ = im_resized.shape

        all_boxes = list()
        while min(current_height, current_width) > net_size:
            feed_imgs = []
            image_tensor = convert_image_to_tensor(im_resized)
            feed_imgs.append(image_tensor)
            feed_imgs = torch.stack(feed_imgs)
            feed_imgs = Variable(feed_imgs)

            feed_imgs = feed_imgs.to(self.device)

            cls_map, reg = self.pnet_detector(feed_imgs)

            cls_map_np = convert_chw_tensor_to_hwc_numpy(cls_map.cpu())
            reg_np = convert_chw_tensor_to_hwc_numpy(reg.cpu())

            boxes = generate_bounding_box(cls_map_np[0, :, :], reg_np, current_scale, self.threshold[0])

            current_scale *= self.scale_factor
            im_resized = resize_image(im, current_scale)
            current_height, current_width, _ = im_resized.shape

            if boxes.size == 0:
                continue
            keep = nms(boxes[:, :5], 0.5, 'Union')
            boxes = boxes[keep]
            all_boxes.append(boxes)

        if len(all_boxes) == 0:
            return None, None

        all_boxes = np.vstack(all_boxes)

        keep = nms(all_boxes[:, 0:5], 0.7, 'Union')
        all_boxes = all_boxes[keep]

        bw = all_boxes[:, 2] - all_boxes[:, 0] + 1
        bh = all_boxes[:, 3] - all_boxes[:, 1] + 1

        boxes = np.vstack([all_boxes[:, 0],
                           all_boxes[:, 1],
                           all_boxes[:, 2],
                           all_boxes[:, 3],
                           all_boxes[:, 4]
                           ])

        boxes = boxes.T

        align_topx = all_boxes[:, 0] + all_boxes[:, 5] * bw
        align_topy = all_boxes[:, 1] + all_boxes[:, 6] * bh
        align_bottomx = all_boxes[:, 2] + all_boxes[:, 7] * bw
        align_bottomy = all_boxes[:, 3] + all_boxes[:, 8] * bh

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 all_boxes[:, 4]
                                 ])
        boxes_align = boxes_align.T

        return boxes, boxes_align

    def detect_rnet(self, im, dets):
        h, w, c = im.shape

        if dets is None:
            return None, None

        dets = get_square_bbox(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            crop_im = cv2.resize(tmp, (24, 24))
            crop_im_tensor = convert_image_to_tensor(crop_im)
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = Variable(torch.stack(cropped_ims_tensors))

        feed_imgs = feed_imgs.to(self.device)

        cls_map, reg = self.rnet_detector(feed_imgs)

        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.threshold[1])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
        else:
            return None, None

        keep = nms(boxes, 0.7)

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        boxes = np.vstack([keep_boxes[:, 0],
                           keep_boxes[:, 1],
                           keep_boxes[:, 2],
                           keep_boxes[:, 3],
                           keep_cls[:, 0]
                           ])

        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0]
                                 ])

        boxes = boxes.T
        boxes_align = boxes_align.T

        return boxes, boxes_align

    def detect_onet(self, im, dets):
        h, w, c = im.shape

        if dets is None:
            return None, None

        dets = get_square_bbox(dets)
        dets[:, 0:4] = np.round(dets[:, 0:4])

        [dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph] = pad(dets, w, h)
        num_boxes = dets.shape[0]

        cropped_ims_tensors = []
        for i in range(num_boxes):
            tmp = np.zeros((tmph[i], tmpw[i], 3), dtype=np.uint8)
            tmp[dy[i]:edy[i] + 1, dx[i]:edx[i] + 1, :] = im[y[i]:ey[i] + 1, x[i]:ex[i] + 1, :]
            crop_im = cv2.resize(tmp, (48, 48))
            crop_im_tensor = convert_image_to_tensor(crop_im)
            cropped_ims_tensors.append(crop_im_tensor)
        feed_imgs = Variable(torch.stack(cropped_ims_tensors))

        feed_imgs = feed_imgs.to(self.device)

        cls_map, reg, landmark = self.onet_detector(feed_imgs)

        cls_map = cls_map.cpu().data.numpy()
        reg = reg.cpu().data.numpy()
        landmark = landmark.cpu().data.numpy()

        keep_inds = np.where(cls_map > self.threshold[2])[0]

        if len(keep_inds) > 0:
            boxes = dets[keep_inds]
            cls = cls_map[keep_inds]
            reg = reg[keep_inds]
            landmark = landmark[keep_inds]
        else:
            return None, None

        keep = nms(boxes, 0.7, mode="Minimum")

        if len(keep) == 0:
            return None, None

        keep_cls = cls[keep]
        keep_boxes = boxes[keep]
        keep_reg = reg[keep]
        keep_landmark = landmark[keep]

        bw = keep_boxes[:, 2] - keep_boxes[:, 0] + 1
        bh = keep_boxes[:, 3] - keep_boxes[:, 1] + 1

        align_topx = keep_boxes[:, 0] + keep_reg[:, 0] * bw
        align_topy = keep_boxes[:, 1] + keep_reg[:, 1] * bh
        align_bottomx = keep_boxes[:, 2] + keep_reg[:, 2] * bw
        align_bottomy = keep_boxes[:, 3] + keep_reg[:, 3] * bh

        align_landmark_topx = keep_boxes[:, 0]
        align_landmark_topy = keep_boxes[:, 1]

        boxes_align = np.vstack([align_topx,
                                 align_topy,
                                 align_bottomx,
                                 align_bottomy,
                                 keep_cls[:, 0]
                                 ])

        boxes_align = boxes_align.T

        landmark = np.vstack([
            align_landmark_topx + keep_landmark[:, 0] * bw,
            align_landmark_topy + keep_landmark[:, 1] * bh,
            align_landmark_topx + keep_landmark[:, 2] * bw,
            align_landmark_topy + keep_landmark[:, 3] * bh,
            align_landmark_topx + keep_landmark[:, 4] * bw,
            align_landmark_topy + keep_landmark[:, 5] * bh,
            align_landmark_topx + keep_landmark[:, 6] * bw,
            align_landmark_topy + keep_landmark[:, 7] * bh,
            align_landmark_topx + keep_landmark[:, 8] * bw,
            align_landmark_topy + keep_landmark[:, 9] * bh,
        ])

        landmark_align = landmark.T

        return boxes_align, landmark_align

    def detect_face(self, img):
        boxes_align = np.array([])
        landmark_align = np.array([])

        if self.pnet_detector:
            boxes, boxes_align = self.detect_pnet(img)
            if boxes_align is None:
                return np.array([]), np.array([])

        if self.rnet_detector:
            boxes, boxes_align = self.detect_rnet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

        if self.onet_detector:
            boxes_align, landmark_align = self.detect_onet(img, boxes_align)
            if boxes_align is None:
                return np.array([]), np.array([])

        return boxes_align, landmark_align
