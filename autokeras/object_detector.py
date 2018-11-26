# ----------------------------------
# The idea of the classes and functions in this file is largely borrowed from
# https://github.com/amdegroot/ssd.pytorch
# A huge thank you to the authors: Max deGroot and Ellis Brown
# ----------------------------------

from __future__ import print_function
from autokeras.object_detection.data import *
from autokeras.object_detection.data import VOC_CLASSES as labelmap
from autokeras.object_detection.data import base_transform
from autokeras.object_detection.utils.augmentations import SSDAugmentation
from autokeras.object_detection.layers.modules import MultiBoxLoss
from autokeras.object_detection.ssd import build_ssd
from autokeras.utils import download_file
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import pickle
import cv2
from matplotlib import pyplot as plt

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

def xavier(param):
    init.xavier_uniform(param)

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()
    

class Timer(object):
    """A simple timer."""
    def __init__(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.average_time = 0.

    def tic(self):
        # using time.time instead of time.clock because time time.clock
        # does not normalize for multithreading
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.average_time = self.total_time / self.calls
        if average:
            return self.average_time
        else:
            return self.diff


class ObjectDetector():
    def __init__(self, cuda=False):
        self.cuda = cuda
        self.net = None

        if torch.cuda.is_available():
            if self.cuda:
                torch.set_default_tensor_type('torch.cuda.FloatTensor')
            if not self.cuda:
                print("WARNING: It looks like you have a CUDA device, but aren't " +
                      "using CUDA.\nRun with --cuda for optimal training speed.")
                torch.set_default_tensor_type('torch.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

    def load(self, trained_model=os.getcwd()+'/object_detection/weights/ssd300_mAP_77.43_v2.pth', trained_model_device='gpu'):
        # https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth
        if trained_model is None:
            file_link = "https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth"
            file_path = "./" + file_link.split('/')[-1]
            trained_model = os.getcwd() + "/" + file_link.split('/')[-1]
            download_file(file_link, file_path)
        # load net
        num_classes = len(labelmap) + 1                      # +1 for background
        self.net = build_ssd('test', 300, num_classes)            # initialize SSD
        if trained_model_device not in ['cpu', 'gpu']:
            raise ValueError("trained_model_device must be either 'cpu' or 'gpu'!")
            exit(0)
        if trained_model_device == 'gpu' and self.cuda is False:
            self.net.load_state_dict(torch.load(trained_model, map_location=lambda storage, loc: storage))
        else:
            self.net.load_state_dict(torch.load(trained_model))
        self.net.eval()
        print('Finished loading model!')

        if self.cuda:
            self.net = self.net.cuda()
            cudnn.benchmark = True
        
    def fit(self, num_classes=1, dataset_format='VOC', dataset=None, dataset_root=VOC_ROOT, train_test_split=True, basenet='vgg16_reducedfc.pth', batch_size=32, resume=None, start_iter=0, num_workers=4, lr=1e-4, momentum=0.9, weight_decay=5e-4, gamma=0.1, save_folder=os.getcwd() + '/object_detection/weights/'):

        if not os.path.exists(save_folder): os.mkdir(save_folder)

        if dataset == 'COCO':
            cfg = coco
            dataset = COCODetection(root=dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        elif dataset == 'VOC':
            cfg = voc
            dataset = VOCDetection(root=dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        elif dataset_format == 'COCO':
            cfg = coco
            if num_classes < 2:
                raise ValueError("num_classes must be equal or greater than 2!")
            cfg['num_classes'] = num_classes + 1
            # TODO
            dataset = COCODetection(root=dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        elif dataset_format == 'VOC':
            cfg = voc
            if num_classes < 1:
                raise ValueError("num_classes must be equal or greater than 2!")
            cfg['num_classes'] = num_classes + 1
            # TODO use MEANS of customized dataset
            dataset = VOC_Custom(root=dataset_root, transform=SSDAugmentation(cfg['min_dim'], MEANS))
        else:
            raise ValueError("ERROR: dataset_format must be one of 'VOC' or 'COCO'!")
        print("dataset length", len(dataset))
    
        print(cfg['num_classes'])
        ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'])
        net = ssd_net
    
        if self.cuda:
            net = torch.nn.DataParallel(ssd_net)
            net = net.cuda()
            cudnn.benchmark = True
    
        if resume:
            print('Resuming training, loading {}...'.format(resume))
            ssd_net.load_weights(resume)
        elif basenet:
            vgg_weights = torch.load(save_folder + basenet)
            print('Loading base network...')
            ssd_net.vgg.load_state_dict(vgg_weights)
        else:
            print('Initializing weights...')
            # initialize newly added layers' weights with xavier method
            ssd_net.extras.apply(weights_init)
            ssd_net.loc.apply(weights_init)
            ssd_net.conf.apply(weights_init)
    
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum,
                              weight_decay=weight_decay)
        criterion = MultiBoxLoss(cfg['num_classes'], 0.5, True, 0, True, 3, 0.5,
                                 False, self.cuda)
    
        net.train()
        # loss counters
        loc_loss = 0
        conf_loss = 0
        epoch = 0
        print('Loading the dataset...')
    
        epoch_size = len(dataset) // batch_size
        # print('Training SSD on:', dataset.name)
        # print('Using the specified args:')
        # print(args)
    
        step_index = 0
    
        data_loader = data.DataLoader(dataset, batch_size,
                                      num_workers=num_workers,
                                      shuffle=True, collate_fn=detection_collate,
                                      pin_memory=False)
        # create batch iterator
        batch_iterator = iter(data_loader)
        for iteration in range(start_iter, cfg['max_iter']):
            if iteration in cfg['lr_steps']:
                step_index += 1
                self.adjust_learning_rate(optimizer, gamma, step_index, lr)
    
            # load train data
            try:
                images, targets = next(batch_iterator)
            except StopIteration:
                batch_iterator = iter(data_loader)
                images, targets = next(batch_iterator)
    
            if self.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True) for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]
            # forward
            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            loc_loss += loss_l.item()
            conf_loss += loss_c.item()
    
            if iteration % 10 == 0:
                print('timer: %.4f sec.' % (t1 - t0))
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % (loss.item()), end=' ')
    
            if iteration != 0 and iteration % 5000 == 0:
                print('Saving state, iter:', iteration)
                torch.save(ssd_net.state_dict(), 'weights/ssd300_VOC_' + repr(iteration) + '.pth')
        # torch.save(ssd_net.state_dict(), save_folder + '' + dataset + '.pth')
        torch.save(ssd_net.state_dict(), save_folder + '' + "ssd300_VOC_" + '.pth')
    
    
    def adjust_learning_rate(self, optimizer, gamma, step, lr):
        """Sets the learning rate to the initial LR decayed by 10 at every
            specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        lr = lr * (gamma ** (step))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def predict(self, path=None, top_k=10):
        from matplotlib.ticker import NullLocator

        dataset_mean = (104, 117, 123)

        image = cv2.imread(path, cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        x = base_transform(rgb_image, 300, dataset_mean)
        x = x.astype(np.float32)
        x = torch.from_numpy(x).permute(2, 0, 1)
        xx = Variable(x.unsqueeze(0))     # wrap tensor in Variable
        if torch.cuda.is_available():
            xx = xx.cuda()
        y = self.net(xx)

        # plt.figure(figsize=(10,10))
        colors = plt.cm.hsv(np.linspace(0, 1, 21)).tolist()
        plt.imshow(rgb_image) # plot the image for matplotlib
        currentAxis = plt.gca()
        currentAxis.set_axis_off()
        currentAxis.xaxis.set_major_locator(NullLocator())
        currentAxis.yaxis.set_major_locator(NullLocator())
        
        detections = y.data
        # scale each detection back up to the image
        scale = torch.Tensor(rgb_image.shape[1::-1]).repeat(2)
        for i in range(detections.size(1)):
            j = 0
            while detections[0,i,j,0] >= 0.6:
                score = detections[0,i,j,0]
                label_name = labelmap[i-1]
                display_txt = '%s: %.2f'%(label_name, score)
                pt = (detections[0,i,j,1:]*scale).cpu().numpy()
                coords = (pt[0], pt[1]), pt[2]-pt[0]+1, pt[3]-pt[1]+1
                color = colors[i]
                currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor=color, linewidth=2))
                currentAxis.text(pt[0], pt[1], display_txt, bbox={'facecolor':color, 'alpha':0.5})
                j+=1
        save_name = path.split('.')
        save_name = '.'.join(save_name[:-1]) + "_prediction." + save_name[-1]
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_name, bbox_inches='tight', pad_inches=0)
        plt.clf()

    
    def evaluate(self, trained_model=os.getcwd()+'/object_detection/weights/ssd300_mAP_77.43_v2.pth', trained_model_device='cpu', save_folder=os.getcwd()+'/object_detection/eval/', confidence_threshold=0.01, top_k=5, voc_root=VOC_ROOT, cleanup=False):
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)

        annopath = os.path.join(voc_root, 'VOC2007', 'Annotations', '%s.xml')
        imgpath = os.path.join(voc_root, 'VOC2007', 'JPEGImages', '%s.jpg')
        imgsetpath = os.path.join(voc_root, 'VOC2007', 'ImageSets', 'Main', '{:s}.txt')
        YEAR = '2007'
        devkit_path = voc_root + 'VOC' + YEAR
        dataset_mean = (104, 117, 123)
        set_type = 'test'

        # load net
        num_classes = len(labelmap) + 1                      # +1 for background
        net = build_ssd('test', 300, num_classes)            # initialize SSD
        if trained_model_device not in ['cpu', 'gpu']:
            raise ValueError("trained_model_device must be either 'cpu' or 'gpu'!")
            exit(0)
        if trained_model_device == 'gpu' and self.cuda is False:
            net.load_state_dict(torch.load(trained_model, map_location=lambda storage, loc: storage))
        else:
            net.load_state_dict(torch.load(trained_model))
        net.eval()
        print('Finished loading model!')
        # load data
        dataset = VOCDetection(voc_root, [('2007', set_type)],
                               BaseTransform(300, dataset_mean),
                               VOCAnnotationTransform())
        if self.cuda:
            net = net.cuda()
            cudnn.benchmark = True
        # evaluation
        self.test_net(annopath, devkit_path, imgsetpath, save_folder, net, self.cuda, dataset,
                 BaseTransform(net.size, dataset_mean), top_k, set_type, 300,
                 thresh=confidence_threshold)
    
    def test_net(self, annopath, devkit_path, imgsetpath, save_folder, net, cuda, dataset, transform, top_k, set_type, 
                 im_size=300, thresh=0.05):
        num_images = len(dataset)
        # all detections are collected into:
        #    all_boxes[cls][image] = N x 5 array of detections in
        #    (x1, y1, x2, y2, score)
        all_boxes = [[[] for _ in range(num_images)]
                     for _ in range(len(labelmap)+1)]
    
        # timers
        _t = {'im_detect': Timer(), 'misc': Timer()}
        output_dir = self.get_output_dir('ssd300_120000', set_type)
        det_file = os.path.join(output_dir, 'detections.pkl')
    
        for i in range(num_images):
            im, gt, h, w = dataset.pull_item(i)
    
            x = Variable(im.unsqueeze(0))
            if self.cuda:
                x = x.cuda()
            _t['im_detect'].tic()
            detections = net(x).data
            detect_time = _t['im_detect'].toc(average=False)
    
            # skip j = 0, because it's the background class
            for j in range(1, detections.size(1)):
                dets = detections[0, j, :]
                mask = dets[:, 0].gt(0.).expand(5, dets.size(0)).t()
                dets = torch.masked_select(dets, mask).view(-1, 5)
                # if dets.dim() == 0:
                if dets.size(0) == 0:
                    continue
                boxes = dets[:, 1:]
                boxes[:, 0] *= w
                boxes[:, 2] *= w
                boxes[:, 1] *= h
                boxes[:, 3] *= h
                scores = dets[:, 0].cpu().numpy()
                cls_dets = np.hstack((boxes.cpu().numpy(),
                                      scores[:, np.newaxis])).astype(np.float32,
                                                                     copy=False)
                all_boxes[j][i] = cls_dets
    
            print('im_detect: {:d}/{:d} {:.3f}s'.format(i + 1,
                                                        num_images, detect_time))
    
        with open(det_file, 'wb') as f:
            pickle.dump(all_boxes, f, pickle.HIGHEST_PROTOCOL)
    
        print('Evaluating detections')
        self.evaluate_detections(annopath, all_boxes, devkit_path, imgsetpath, output_dir, dataset, set_type)
    
    
    def evaluate_detections(self, annopath, box_list, devkit_path, imgsetpath, output_dir, dataset, set_type):
        self.write_voc_results_file(box_list, dataset, devkit_path, set_type)
        self.do_python_eval(annopath, devkit_path, imgsetpath, set_type, output_dir)
    

    def write_voc_results_file(self, all_boxes, dataset, devkit_path, set_type):
        for cls_ind, cls in enumerate(labelmap):
            print('Writing {:s} VOC results file'.format(cls))
            filename = self.get_voc_results_file_template(devkit_path, set_type, cls)
            with open(filename, 'wt') as f:
                for im_ind, index in enumerate(dataset.ids):
                    dets = all_boxes[cls_ind+1][im_ind]
                    if dets == []:
                        continue
                    # the VOCdevkit expects 1-based indices
                    for k in range(dets.shape[0]):
                        f.write('{:s} {:.3f} {:.1f} {:.1f} {:.1f} {:.1f}\n'.
                                format(index[1], dets[k, -1],
                                       dets[k, 0] + 1, dets[k, 1] + 1,
                                       dets[k, 2] + 1, dets[k, 3] + 1))

    
    def do_python_eval(self, annopath, devkit_path, imgsetpath, set_type, output_dir='output', use_07=True):
        cachedir = os.path.join(devkit_path, 'annotations_cache')
        aps = []
        # The PASCAL VOC metric changed in 2010
        use_07_metric = use_07
        print('VOC07 metric? ' + ('Yes' if use_07_metric else 'No'))
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
        for i, cls in enumerate(labelmap):
            filename = self.get_voc_results_file_template(devkit_path, set_type, cls)
            rec, prec, ap = self.voc_eval(
               filename, annopath, imgsetpath.format(set_type), cls, cachedir,
               ovthresh=0.5, use_07_metric=use_07_metric)
            aps += [ap]
            print('AP for {} = {:.4f}'.format(cls, ap))
            with open(os.path.join(output_dir, cls + '_pr.pkl'), 'wb') as f:
                pickle.dump({'rec': rec, 'prec': prec, 'ap': ap}, f)
        print('Mean AP = {:.4f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('Results:')
        for ap in aps:
            print('{:.3f}'.format(ap))
        print('{:.3f}'.format(np.mean(aps)))
        print('~~~~~~~~')
        print('')
        print('--------------------------------------------------------------')
        print('Results computed with the **unofficial** Python eval code.')
        print('Results should be very close to the official MATLAB eval code.')
        print('--------------------------------------------------------------')
    
    
    def get_voc_results_file_template(self, devkit_path, image_set, cls):
        # VOCdevkit/VOC2007/results/det_test_aeroplane.txt
        filename = 'det_' + image_set + '_%s.txt' % (cls)
        filedir = os.path.join(devkit_path, 'results')
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        path = os.path.join(filedir, filename)
        return path
    
    
    def voc_eval(self, detpath,
                 annopath,
                 imagesetfile,
                 classname,
                 cachedir,
                 ovthresh=0.5,
                 use_07_metric=True):
        """rec, prec, ap = voc_eval(detpath,
                               annopath,
                               imagesetfile,
                               classname,
                               [ovthresh],
                               [use_07_metric])
    Top level function that does the PASCAL VOC evaluation.
    detpath: Path to detections
       detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
       annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.5)
    [use_07_metric]: Whether to use VOC07's 11 point AP computation
       (default True)
    """
    # assumes detections are in detpath.format(classname)
    # assumes annotations are in annopath.format(imagename)
    # assumes imagesetfile is a text file with each line an image name
    # cachedir caches the annotations in a pickle file
    # first load gt
        if not os.path.isdir(cachedir):
            os.mkdir(cachedir)
        cachefile = os.path.join(cachedir, 'annots.pkl')
        # read list of images
        with open(imagesetfile, 'r') as f:
            lines = f.readlines()
        imagenames = [x.strip() for x in lines]
        if not os.path.isfile(cachefile):
            # load annots
            recs = {}
            for i, imagename in enumerate(imagenames):
                recs[imagename] = self.parse_rec(annopath % (imagename))
                if i % 100 == 0:
                    print('Reading annotation for {:d}/{:d}'.format(
                       i + 1, len(imagenames)))
            # save
            print('Saving cached annotations to {:s}'.format(cachefile))
            with open(cachefile, 'wb') as f:
                pickle.dump(recs, f)
        else:
            # load
            with open(cachefile, 'rb') as f:
                recs = pickle.load(f)
    
        # extract gt objects for this class
        class_recs = {}
        npos = 0
        for imagename in imagenames:
            R = [obj for obj in recs[imagename] if obj['name'] == classname]
            bbox = np.array([x['bbox'] for x in R])
            difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
            det = [False] * len(R)
            npos = npos + sum(~difficult)
            class_recs[imagename] = {'bbox': bbox,
                                     'difficult': difficult,
                                     'det': det}
    
        # read dets
        detfile = detpath.format(classname)
        with open(detfile, 'r') as f:
            lines = f.readlines()
        if any(lines) == 1:
    
            splitlines = [x.strip().split(' ') for x in lines]
            image_ids = [x[0] for x in splitlines]
            confidence = np.array([float(x[1]) for x in splitlines])
            BB = np.array([[float(z) for z in x[2:]] for x in splitlines])
    
            # sort by confidence
            sorted_ind = np.argsort(-confidence)
            sorted_scores = np.sort(-confidence)
            BB = BB[sorted_ind, :]
            image_ids = [image_ids[x] for x in sorted_ind]
    
            # go down dets and mark TPs and FPs
            nd = len(image_ids)
            tp = np.zeros(nd)
            fp = np.zeros(nd)
            for d in range(nd):
                R = class_recs[image_ids[d]]
                bb = BB[d, :].astype(float)
                ovmax = -np.inf
                BBGT = R['bbox'].astype(float)
                if BBGT.size > 0:
                    # compute overlaps
                    # intersection
                    ixmin = np.maximum(BBGT[:, 0], bb[0])
                    iymin = np.maximum(BBGT[:, 1], bb[1])
                    ixmax = np.minimum(BBGT[:, 2], bb[2])
                    iymax = np.minimum(BBGT[:, 3], bb[3])
                    iw = np.maximum(ixmax - ixmin, 0.)
                    ih = np.maximum(iymax - iymin, 0.)
                    inters = iw * ih
                    uni = ((bb[2] - bb[0]) * (bb[3] - bb[1]) +
                           (BBGT[:, 2] - BBGT[:, 0]) *
                           (BBGT[:, 3] - BBGT[:, 1]) - inters)
                    overlaps = inters / uni
                    ovmax = np.max(overlaps)
                    jmax = np.argmax(overlaps)
    
                if ovmax > ovthresh:
                    if not R['difficult'][jmax]:
                        if not R['det'][jmax]:
                            tp[d] = 1.
                            R['det'][jmax] = 1
                        else:
                            fp[d] = 1.
                else:
                    fp[d] = 1.
    
            # compute precision recall
            fp = np.cumsum(fp)
            tp = np.cumsum(tp)
            rec = tp / float(npos)
            # avoid divide by zero in case the first detection matches a difficult
            # ground truth
            prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
            ap = self.voc_ap(rec, prec, use_07_metric)
        else:
            rec = -1.
            prec = -1.
            ap = -1.
    
        return rec, prec, ap
    

    def parse_rec(self, filename):
        """ Parse a PASCAL VOC xml file """
        tree = ET.parse(filename)
        objects = []
        for obj in tree.findall('object'):
            obj_struct = {}
            obj_struct['name'] = obj.find('name').text
            obj_struct['pose'] = obj.find('pose').text
            obj_struct['truncated'] = int(obj.find('truncated').text)
            obj_struct['difficult'] = int(obj.find('difficult').text)
            bbox = obj.find('bndbox')
            obj_struct['bbox'] = [int(bbox.find('xmin').text) - 1,
                                  int(bbox.find('ymin').text) - 1,
                                  int(bbox.find('xmax').text) - 1,
                                  int(bbox.find('ymax').text) - 1]
            objects.append(obj_struct)
    
        return objects
    
    
    def voc_ap(self, rec, prec, use_07_metric=True):
        """ ap = voc_ap(rec, prec, [use_07_metric])
        Compute VOC AP given precision and recall.
        If use_07_metric is true, uses the
        VOC 07 11 point method (default:True).
        """
        if use_07_metric:
            # 11 point metric
            ap = 0.
            for t in np.arange(0., 1.1, 0.1):
                if np.sum(rec >= t) == 0:
                    p = 0
                else:
                    p = np.max(prec[rec >= t])
                ap = ap + p / 11.
        else:
            # correct AP calculation
            # first append sentinel values at the end
            mrec = np.concatenate(([0.], rec, [1.]))
            mpre = np.concatenate(([0.], prec, [0.]))
    
            # compute the precision envelope
            for i in range(mpre.size - 1, 0, -1):
                mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    
            # to calculate area under PR curve, look for points
            # where X axis (recall) changes value
            i = np.where(mrec[1:] != mrec[:-1])[0]
    
            # and sum (\Delta recall) * prec
            ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap
    
    
    def get_output_dir(self, name, phase):
        """Return the directory where experimental artifacts are placed.
        If the directory does not exist, it is created.
        A canonical path is built using the name from an imdb and a network
        (if not None).
        """
        filedir = os.path.join(name, phase)
        if not os.path.exists(filedir):
            os.makedirs(filedir)
        return filedir
    
