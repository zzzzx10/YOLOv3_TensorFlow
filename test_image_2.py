# coding: utf-8
"""
Use pb file to do predict
Author: zzzzx10
Date : 2019-07-08 16:36:45

"""
from __future__ import division, print_function

import tensorflow as tf
import numpy as np
import cv2

from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms, cpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from timeit import default_timer as timer

from model import yolov3

anchor_path = "./data/yolo_anchors.txt"
class_name_path = "./data/coco.names"
anchors = parse_anchors(anchor_path)
classes = read_class_names(class_name_path)
num_class = len(classes)
new_size = [416,416]
color_table = get_color_table(num_class)


score_threshold = 0.4
iou_threshold = 0.5
pb_file = "./data/darknet_weights/yolov3.pb"



with tf.gfile.FastGFile(pb_file, 'rb') as f:
    frozen_graph_def = tf.GraphDef()
    frozen_graph_def.ParseFromString(f.read())


input_node_names = ["Placeholder:0"]
output_node_names = ["yolov3/yolov3_head/feature_map_1:0",
                     "yolov3/yolov3_head/feature_map_2:0",
                     'yolov3/yolov3_head/feature_map_3:0']
return_elements = input_node_names + output_node_names


graph = tf.Graph()
with graph.as_default():
    return_elements = tf.import_graph_def(frozen_graph_def, return_elements=return_elements)

input_tensor, output_tensors = return_elements[0], return_elements[1:]

yolo_model = yolov3(num_class, anchors)

#每一个要做的



with tf.Session(graph=graph) as sess:

    for i in range(5):
        time1 = timer()
        input_image = str(i)+".jpg"
        img_ori = cv2.imread(input_image)
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(new_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = np.asarray(img, np.float32)
        img = img[np.newaxis, :] / 255.


        feature_map = sess.run(output_tensors, feed_dict={input_tensor: img})

        feature_map_1, feature_map_2, feature_map_3 = feature_map
        feature_map_1 = tf.constant(feature_map_1, dtype=tf.float32)
        feature_map_2 = tf.constant(feature_map_2, dtype=tf.float32)
        feature_map_3 = tf.constant(feature_map_3, dtype=tf.float32)
        tf_img_size = tf.constant(value=new_size, dtype=tf.int32)


        #with tf.Session(graph=graph) as sess:
        pred_boxes, pred_confs, pred_probs = yolo_model.predict2(feature_map_1, feature_map_2, feature_map_3,tf_img_size)
        pred_scores = pred_confs * pred_probs

        #GPU
        #boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, num_class, max_boxes=200, score_thresh=0.3, nms_thresh=0.45)
        #boxes_, scores_, labels_ = sess.run([boxes, scores, labels])

        #CPU
        boxes, scores = sess.run([pred_boxes, pred_scores], feed_dict={input_tensor: img})
        boxes_, scores_, labels_ = cpu_nms(boxes, scores, num_class, score_thresh=0.4, iou_thresh=0.5)

        boxes_[:, 0] *= (width_ori / float(new_size[0]))
        boxes_[:, 2] *= (width_ori / float(new_size[0]))
        boxes_[:, 1] *= (height_ori / float(new_size[1]))
        boxes_[:, 3] *= (height_ori / float(new_size[1]))



        for i in range(len(boxes_)):
            x0, y0, x1, y1 = boxes_[i]
            scores = scores_[i]
            plot_one_box(img_ori, [x0, y0, x1, y1], label=classes[labels_[i]]+':'+str(scores)[:6], color=color_table[labels_[i]])
        # cv2.imshow('Detection result', img_ori)
        # cv2.imwrite('detection_result.jpg', img_ori)
        cv2.waitKey(30)

        time2 = timer()
        print(time2-time1)





'''
#我需要这样读数据！
# Load Frozen Network Model & Broadcast to Worker Nodes
with tf.gfile.FastGFile(model_file, 'rb') as f:
    model_data = f.read()

# Load Graph Definition
graph_def = tf.GraphDef()
graph_def.ParseFromString(model_data.value)
'''
