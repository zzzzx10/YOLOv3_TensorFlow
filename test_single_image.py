# coding: utf-8

from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import argparse
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms, cpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize
from timeit import default_timer as timer

from model import yolov3

parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
#parser.add_argument("input_image", type=str,
#                    help="The path of the input image.")
parser.add_argument("--anchor_path", type=str, default="./data/yolo_anchors.txt",
                    help="The path of the anchor txt file.")
parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                    help="Resize the input image with `new_size`, size format: [width, height]")
parser.add_argument("--class_name_path", type=str, default="./data/coco.names",
                    help="The path of the class names.")
parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3.ckpt",
                    help="The path of the weights to restore.")
args = parser.parse_args()

args.anchors = parse_anchors(args.anchor_path)
args.classes = read_class_names(args.class_name_path)
args.num_class = len(args.classes)

color_table = get_color_table(args.num_class)

input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3], name='input_data')
yolo_model = yolov3(args.num_class, args.anchors)
with tf.variable_scope('yolov3'):
    pred_feature_maps = yolo_model.forward(input_data, False)
pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

pred_scores = pred_confs * pred_probs
saver = tf.train.Saver()

for y in range(5):
    time1 = timer()
    input_image = str(y)+".jpg"
    img_ori = cv2.imread(input_image)
    height_ori, width_ori = img_ori.shape[:2]
    img = cv2.resize(img_ori, tuple(args.new_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    with tf.Session() as sess:
        saver.restore(sess, args.restore_path)


        #如果用cpu 注释下一句
        #boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)





        # CPU
        boxes, scores = sess.run([pred_boxes, pred_scores], feed_dict={input_data: img})
        boxes_, scores_, labels_ = cpu_nms(boxes, scores, args.num_class, score_thresh=0.4, iou_thresh=0.5)

        #GPU
        #boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})

        boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
        boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))

        print("box coords:")
        print(boxes_)
        print('*' * 30)
        print("scores:")
        print(scores_)
        print('*' * 30)
        print("labels:")
        print(labels_)


        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], label=args.classes[labels_[i]] + ', {:.2f}%'.format(scores_[i] * 100), color=color_table[labels_[i]])
        cv2.imshow('Detection result', img_ori)
        #cv2.imwrite('detection_result.jpg', img_ori)
        time2 = timer()
        print(time2-time1)
        cv2.waitKey(0)

