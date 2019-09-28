#!/usr/bin/env python
# -*- coding: utf-8 -*-
# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------
# 根据自己的数据集训练好模型后，要想运行demo.py批量处理测试图片，
# 并按照<image_id> <class_id> <confidence> <xmin> <ymin> <xmax> <ymax>格式输出信息，
# 需要按照https://www.jianshu.com/p/8bbc0af6fc91 进行修改
# ----------------------------------------------------------------------------------
"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms

from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True) # 不使用科学计数输出
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_70000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

# 画出测试图片的bounding boxes, 参数im为测试图片, dets为非极大值抑制后的bbox和score的数组
# thresh是最后score的阈值，高于该阈值的候选框才会被画出来
def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    # np.where(condition, x, y)，满足条件(condition)，输出x，不满足输出y
    # np.where(condition)，只有条件 (condition)，没有x和y，则输出满足条件(即非0)元素的坐标
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    # python-opencv 中读取图片默认保存为[w,h,channel](w,h顺序不确定)
    # 其中 channel：BGR 存储，而画图时，需要按RGB格式，因此此处作转换
    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:  # 从dets中取出 bbox, score
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)# 线宽
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

# 对测试图片提取预选框，并进行非极大值抑制,然后调用上个函数画出矩形框
def demo(sess, net, image_name):
    # 参数：net 测试时使用的网络结构；image_name:图片名称
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    # from model.test import im_detect
    # def im_detect(sess, net, im): return scores, pred_boxes
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8  # score 阈值，最后画出候选框时需要，>thresh才会被画出
    NMS_THRESH = 0.3   # 非极大值抑制的阈值，剔除重复候选框
    for cls_ind, cls in enumerate(CLASSES[1:]):  # 获得CLASSES中 类别的下标cls_ind和类别名cls
        cls_ind += 1 # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]  
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
                        # 将bbox,score 一起存入dets
        keep = nms(dets, NMS_THRESH)  # 进行非极大值抑制，得到抑制后的 dets
        dets = dets[keep, :]
        vis_detections(im, cls, dets, thresh=CONF_THRESH)

def parse_args():  # 解析命令行参数
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()
    # model path
    demonet = args.demo_net
    dataset = args.dataset

#####################################
    demonet = 'vgg16'
    dataset = 'pascal_voc'
#####################################

    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])

#####################################
    tfmodel = '/home/seucar/Desktop/tf-faster-rcnn/output/vgg16/voc_2007_trainval/default/vgg16_faster_rcnn_iter_70000.ckpt'
#####################################

    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config                                           # 通过命令 "with tf.device('/cpu:0'):",允许手动设置操作运行的设备
    tfconfig = tf.ConfigProto(allow_soft_placement=True)   # 如果手动设置的设备不存在或者不可用，设置allow_soft_placement=True
    tfconfig.gpu_options.allow_growth = True               # 允许tf自动选择一个存在并且可用的设备来运行操作
    pass                                                   # allow_growth option，动态申请显存，需要多少申请多少
    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()                      # from nets.vgg16 import vgg16
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)     # from nets.resnet_v1 import resnetv1
    else:
        raise NotImplementedError
    # create_architecture(self, mode, num_classes, tag=None, anchor_scales=(8, 16, 32), anchor_ratios=(0.5, 1, 2)):
    net.create_architecture("TEST", 21, tag='default', anchor_scales=[8, 16, 32])
    # 用自己的数据集测试时，21根据classes类别数量修改，为需要识别的类别+1
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)  # 从 ckpt 恢复
    print('Loaded network {:s}'.format(tfmodel))
    # 测试的图片，保存在tf-faster-rcnn-contest/data/demo 路径
    im_names = ['000456.jpg', '000542.jpg', '001150.jpg',
                '001763.jpg', '004545.jpg']
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        demo(sess, net, im_name)
    plt.show()
