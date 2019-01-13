from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import torch
import cv2
import cPickle
import numpy as np

import network
from wsddn import WSDDN
from utils.timer import Timer
from fast_rcnn.nms_wrapper import nms

from fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from datasets.factory import get_imdb
from fast_rcnn.config import cfg, cfg_from_file, get_output_dir
import pdb

# hyper-parameters
# ------------
imdb_name = 'voc_2007_test'
cfg_file = 'experiments/cfgs/wsddn.yml'
trained_model_fmt = 'models/saved_model/{}_{}.h5'

rand_seed = 1024
save_name = '{}_{}'
max_per_image = 300
thresh = 0.0001
visualize = False

# ------------

if rand_seed is not None:
    np.random.seed(rand_seed)

if rand_seed is not None:
    np.random.seed(rand_seed)

# load config
cfg_from_file(cfg_file)


def vis_detections(im, class_name, dets, thresh=0.8):
    """Visual debugging of detections."""
    for i in range(np.minimum(10, dets.shape[0])):
        bbox = tuple(int(np.round(x)) for x in dets[i, :4])
        score = dets[i, -1]
        if score > thresh:
            cv2.rectangle(im, bbox[0:2], bbox[2:4], (0, 204, 0), 2)
            cv2.putText(im, '%s: %.3f' % (class_name, score), (bbox[0], bbox[1] + 15), cv2.FONT_HERSHEY_PLAIN,
                        1.0, (0, 0, 255), thickness=1)
    return im


def im_detect(net, image, rois):
    """Detect object classes in an image given object proposals.
    Returns:
        scores (ndarray): R x K array of object class scores (K includes
            background as object category 0)
        boxes (ndarray): R x (4*K) array of predicted bounding boxes
    """

    im_data, im_scales = net.get_image_blob(image)
    rois = np.hstack((np.zeros((rois.shape[0],1)),rois*im_scales[0]))
    im_info = np.array(
        [[im_data.shape[1], im_data.shape[2], im_scales[0]]],
        dtype=np.float32)

    cls_prob = net(im_data, rois, im_info)
    
    #pdb.set_trace()

    scores = cls_prob.data.cpu().numpy()
    boxes = rois[:, 1:5] / im_info[0][2]

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = bbox_pred.data.cpu().numpy()
        pred_boxes = bbox_transform_inv(boxes, box_deltas)
        pred_boxes = clip_boxes(pred_boxes, image.shape)
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    return scores, pred_boxes


def test_net(name, net, imdb, max_per_image=300, thresh=0.05, visualize=False,
             logger=None, step=None):
    """Test a Fast R-CNN network on an image database."""


    tf_board_images = []

    num_images = len(imdb.image_index)
    #num_images = 10

    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in xrange(num_images)]
                 for _ in xrange(imdb.num_classes+1)]


    output_dir = get_output_dir(imdb, name)

    # timers
    _t = {'im_detect': Timer(), 'misc': Timer()}
    det_file = os.path.join(output_dir, 'detections.pkl')

    roidb = imdb.roidb



    for i in range(num_images):
        im = cv2.imread(imdb.image_path_at(i))
        rois = imdb.roidb[i]['boxes']
        _t['im_detect'].tic()
        scores, boxes = im_detect(net, im, rois)

        #pdb.set_trace()

        detect_time = _t['im_detect'].toc(average=False)

        _t['misc'].tic()
        if visualize:
            # im2show = np.copy(im[:, :, (2, 1, 0)])
            im2show = np.copy(im)

        # skip j = 0, because it's the background class
        for j in xrange(1, imdb.num_classes+1):
            newj = j-1
            inds = np.where(scores[:, newj] > thresh)[0]
            cls_scores = scores[inds, newj]
            cls_boxes = boxes[inds, newj * 4:(newj + 1) * 4]
            cls_dets = np.hstack((cls_boxes, cls_scores[:, np.newaxis])) \
                .astype(np.float32, copy=False)
            
            #pdb.set_trace()
                
            keep = nms(cls_dets, cfg.TEST.NMS)
            cls_dets = cls_dets[keep, :]
            #if visualize:
            if visualize and np.random.rand()<0.0008:
                #pdb.set_trace()
                im2show = vis_detections(im2show, imdb.classes[newj], cls_dets,thresh = 0.02)
                #changed imdb.classes[j] to imdb.classes[newj]
                tf_board_images.append(im2show[:,:,(2,1,0)])

            all_boxes[j][i] = cls_dets
            # Chnaged all_boxes[j][i] = cls_dets to all_boxes[newj][i] = cls_dets    


        # Limit to max_per_image detections *over all classes*
        if max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][i][:, -1]
                                      for j in xrange(1, imdb.num_classes)])
            if len(image_scores) > max_per_image:
                image_thresh = np.sort(image_scores)[-max_per_image]
                for j in xrange(1, imdb.num_classes):
                    keep = np.where(all_boxes[j][i][:, -1] >= image_thresh)[0]
                    all_boxes[j][i] = all_boxes[j][i][keep, :]
        nms_time = _t['misc'].toc(average=False)

        #if i%10 == 0:
        print('im_detect: {:d}/{:d} {:.3f}s {:.3f}s'.format(i + 1, num_images, detect_time, nms_time))

        #pdb.set_trace()
    logger.image_summary(tag = 'test/image_step_'+str(step), images = tf_board_images, step = step)

        #if visualize and np.random.rand()<0.01:
        #    # TODO: Visualize here using tensorboard
        #    # TODO: use the logger that is an argument to this function
        #    print('Visualizing')

        #    #logger.image_summary(tag = 'test/image_', images = im2show, step = step)
        #    #cv2.imshow('test', im2show)
        #    #cv2.waitKey(1)

    with open(det_file, 'wb') as f:
        cPickle.dump(all_boxes, f, cPickle.HIGHEST_PROTOCOL)

    print('Evaluating detections')
    #pdb.set_trace()

    #print(len(all_boxes))

    aps = imdb.evaluate_detections(all_boxes, output_dir)
    return aps


if __name__ == '__main__':
    # load data
    imdb = get_imdb(imdb_name)
    imdb.competition_mode(on=True)

    # load net
    net = WSDDN(classes=imdb.classes, debug=False)
    trained_model = trained_model_fmt.format(cfg.TRAIN.SNAPSHOT_PREFIX,100000)
    network.load_net(trained_model, net)
    print('load model successfully!')

    net.cuda()
    net.eval()

    # evaluation
    aps = test_net(save_name, net, imdb, 
                   max_per_image, thresh=thresh, visualize=vis)



def test_the_model(net, logger, step):

    imdb = get_imdb(imdb_name)
    imdb.competition_mode(on=True)

    net.eval()
    aps = test_net(save_name, net, imdb, 
                   max_per_image = max_per_image, thresh=0.0001, visualize= True, logger=logger, step=step)

    #aps = test_net(save_name, net, imdb, 
    #               max_per_image = max_per_image, thresh=0.0001, visualize= True, logger=logger, step=step)


    #aps = test_net(save_name, net, imdb, 
    #               max_per_image = max_per_image, thresh=0.005, visualize= True, logger=logger, step=step)

    #aps = test_net('test_log', net, imdb, 
    #               max_per_image = max_per_image, thresh=0.01, visualize= True, logger=logger, step=step)


    net.train()

    return aps