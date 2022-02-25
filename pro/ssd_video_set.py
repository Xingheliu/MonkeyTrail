# -- coding: utf-8 --
import os
import pandas as pd
import cv2
import sys
import time
import argparse
import multiprocessing
import math
#
import collections
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
# from moviepy.editor import ImageSequenceClip
#from IPython.display import HTML
import csv



# print box informations
def output_detection_information(image_np,
                                 boxes,
                                 classes,
                                 scores,
                                 catagory_index,
                                 instance_masks=None,
                                 keypoints=None,
                                 use_normalized_coordinates=False,
                                 max_boxes_to_draw=20,
                                 min_score_thresh=.65,
                                 agnostic_mode=False,
                                 line_thickness=8):
    class_names = []
    scoreis = []
    ymins = []
    xmins = []
    ymaxs = []
    xmaxs = []
    class_name = " "
    scorei = 0
    ymin = 0
    xmin = 0
    ymax = 0
    xmax = 0
    box_to_display_str_map = collections.defaultdict(list)
    box_to_color_map = collections.defaultdict(str)
    box_to_instance_masks_map = {}
    box_to_keypoints_map = collections.defaultdict(list)
    if not max_boxes_to_draw:
        max_boxes_to_draw = boxes.shape[0]
    for i in range(min(max_boxes_to_draw, boxes.shape[0])):
        if scores is None or scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            ymin, xmin, ymax, xmax = box
            if keypoints is not None:
                box_to_keypoints_map[box].extend(keypoints[i])
            if scores is None:
                box_to_color_map[box] = 'black'
            else:
                if not agnostic_mode:
                    if classes[i] in category_index.keys():
                        class_name = category_index[classes[i]]['name']
                    else:
                        class_name = 'N/A'
                    display_str = '{}: {}%'.format(class_name, int(100 * scores[i]))
                    #
                    scorei = scores[i]
                else:
                    display_str = 'score: {}%'.format(int(100 * scores[i]))
                box_to_display_str_map[box].append(display_str)
                class_names.append(class_name)
                scoreis.append(scorei)
                ymins.append(ymin)
                xmins.append(xmin)
                ymaxs.append(ymax)
                xmaxs.append(xmax)
    if (len(class_names) == 0):
        class_names.append(" ")
        scoreis.append(0)
        ymins.append(0)
        xmins.append(0)
        ymaxs.append(0)
        xmaxs.append(0)
    return class_names, scoreis, ymins, xmins, ymaxs, xmaxs


def detect_objects(image_np, sess, detection_graph,frame_num,PATH_TO_OUTPUT_INFO):
    # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
    image_np_expanded = np.expand_dims(image_np, axis=0)
    image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')

    # Each box represents a part of the image where a particular object was detected.
    boxes = detection_graph.get_tensor_by_name('detection_boxes:0')

    # Each score represent how level of confidence for each of the objects.
    # Score is shown on the result image, together with the class label.
    scores = detection_graph.get_tensor_by_name('detection_scores:0')
    classes = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections = detection_graph.get_tensor_by_name('num_detections:0')

    # Actual detection.
    (boxes, scores, classes, num_detections) = sess.run(
        [boxes, scores, classes, num_detections],
        feed_dict={image_tensor: image_np_expanded})

    # Visualization of the results of a detection.
    vis_util.visualize_boxes_and_labels_on_image_array(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    # Write detection information to file
    classname, score, ymin, xmin, ymax, xmax = output_detection_information(
        image_np,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

    ob_num = len(classname)
    nu = 0
    write_end=0
    while nu < ob_num:

        if classname[nu] == 'monkey':
            [height, width, pixels]=image_np.shape
            ymin[nu], xmin[nu], ymax[nu], xmax[nu] = ymin[nu]*height, xmin[nu]*width, ymax[nu]*height, xmax[nu]*width
            with open(PATH_TO_OUTPUT_INFO, 'a', newline='') as f:
                f_csv = csv.writer(f)
                headers = [frame_num, classname[nu], score[nu], xmin[nu], xmax[nu], ymin[nu], ymax[nu]]
                f_csv.writerow(headers)
            write_end=1
            break
        nu = nu + 1

    if write_end == 0:
        with open(PATH_TO_OUTPUT_INFO, 'a', newline='') as f:
            f_csv = csv.writer(f)
            headers = [frame_num, 0, 0, 0, 0, 0, 0]
            f_csv.writerow(headers)

    return image_np


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)

os.environ['CUDA_VISIBLE_DEVICES'] = '/gpu:0'
CWD_PATH = os.getcwd()

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'D:/workproject/PycharmProjects/MonkeyTrail/model/day/frozen_inference_graph.pb'
PATH_TO_CKPT = 'D:/workproject/PycharmProjects/MonkeyTrail/model/day/frozen_inference_graph.pb'
PATH_TO_LABELS = 'D:/workproject/PycharmProjects/MonkeyTrail/model/day/label_map.pbtxt'

NUM_CLASSES = 1
# Loading label map
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
# Load a frozen TF model
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

class SSD_M():
    def __init__(self):
        self.a=1
        self.frame_num = 0
        self.csv_num = 0
        self.PATH_TO_OUTPUT_INFO = './temp/temp2.csv'
        mf = pd.read_csv('./temp/temp1.csv')  # 返回一个DataFrame的对象，这个是pandas的一个数据结构
        mf.columns = ['frame', 'x1', 'x2', 'y1', 'y2']
        xm = mf[['frame', 'x1', 'x2', 'y1', 'y2']]
        self.xm = np.array(xm)
        self.csv_num_all = len(xm)


    def ssd_detect(self,video_input, video_output):
        def process_image(image):
            self.frame_num = self.frame_num + 1
            if self.csv_num >= self.csv_num_all:
                return image
            if self.frame_num == self.xm[self.csv_num][0]:
                with detection_graph.as_default():
                    with tf.Session(graph=detection_graph) as sess:
                        image_process = detect_objects(np.array(image), sess, detection_graph,
                                                       self.frame_num, self.PATH_TO_OUTPUT_INFO)
                self.csv_num = self.csv_num + 1
                return image_process
            else:
                return image

        with open(self.PATH_TO_OUTPUT_INFO, 'w', newline='') as f:
            f_csv = csv.writer(f)
            headers = ['frame_num', 'classname', 'score', 'x1', 'x2', 'y1', 'y2']
            f_csv.writerow(headers)
        clip = VideoFileClip(video_input)
        white_clip = clip.fl_image(process_image)
        white_clip.write_videofile(video_output, codec='libx264', audio=False)
