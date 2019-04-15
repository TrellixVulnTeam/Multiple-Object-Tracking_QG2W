from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os, sys
import time
import math
import cv2
import argparse

sys.path.append('./utils')
from stream import VideoInputStream
sys.path.append('./datasets')
from mask_rcnn_tfrecords import batch_segmentation_masks, visualize_masks

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize mask-rcnn results')
    parser.add_argument(
        '--input_video_path', required=True,
        help='path to the input video stream')
    parser.add_argument(
        '--input_detections_path', required=True,
        help='path to the detections')
    parser.add_argument(
        '--output_video_path', required=True,
        help='path to the output video')
    parser.add_argument(
        '--max_frames', type=int, default=1000,
        help='Number of frames')
    parser.add_argument('--scale_boxes', help='Scale the bounding boxes',
                        action='store_false')
    args = parser.parse_args()

    return args

def get_class_groups_kitchen():
    people_cls = [1]
    bottle_cls = [40]
    wineglass_cls = [41]
    cup_cls = [42]
    fork_cls = [43]
    knife_cls = [44]
    spoon_cls = [45]
    bowl_cls = [46]

    utensils_cls = [ 40, 41, 42, 43, 44, 45, 46]
    electronics_cls = [ 69 , 70, 71, 73 ]

    class_groups = [people_cls, bottle_cls, wineglass_cls, cup_cls, \
                    fork_cls, knife_cls, spoon_cls, bowl_cls]
    class_groups = [ people_cls, utensils_cls, electronics_cls ]

    return class_groups

def get_class_groups_sports():
    people_cls = [1]
    ball_cls = [33]

    class_groups = [people_cls, ball_cls]
    return class_groups

def get_class_groups():
    people_cls = [1]
    twowheeler_cls = [2, 4]
    vehicle_cls = [3, 6, 7, 8]

    #(40, 'bottle')
    #(41, 'wine glass')
    #(42, 'cup')
    #(43, 'fork')
    #(44, 'knife')
    #(45, 'spoon')
    #(46, 'bowl')

    utensils_cls = [40, 41, 42, 43, 44, 45, 46]

    #(14, 'bench')
    #(57, 'chair')
    #(58, 'couch')
    #(61, 'dining table')
    furniture_cls = [14, 57, 58, 61]

    class_groups = [people_cls, twowheeler_cls, vehicle_cls, utensils_cls, furniture_cls]

    return class_groups

def visualize(input_video_path,
              detections_path,
              output_video_path,
              scale_boxes,
              class_groups,
              max_frames = None,
              start_frame = 0):

    detections = {}
    if detections_path:
        detections = np.load(detections_path)[()]

    s = VideoInputStream(input_video_path,
                         start_frame=start_frame)

    curr_pos = start_frame

    vid_out = cv2.VideoWriter(output_video_path,
                              cv2.VideoWriter_fourcc(*'X264'),
                              s.rate, (2*s.width, s.height))

    num_classes = len(class_groups) + 1

    for f in s:
        print('frame', curr_pos)
        if ((max_frames is not None) and
            (curr_pos > start_frame + max_frames)):
            break

        boxes, classes, scores, masks = detections[curr_pos]

        num_objects = np.expand_dims(np.array(scores.shape[0]), axis=0)

        boxes = np.expand_dims(np.array(boxes), axis=0)
        classes = np.expand_dims(np.array(classes), axis=0)
        masks = np.expand_dims(np.array(masks), axis=0)
        scores = np.expand_dims(np.array(scores), axis=0)

        labels, _ = batch_segmentation_masks(1, (s.height, s.width),
                                             boxes, classes, masks,
                                             scores, num_objects, True,
                                             class_groups,
                                             scale_boxes=scale_boxes)

        if vid_out:
            vis_shape = (labels.shape[1], labels.shape[2], 3)
            vis_labels = visualize_masks(labels, 1, vis_shape,
                                         num_classes=num_classes)

            vis_labels = vis_labels[0]

            labels_image = cv2.addWeighted(f, 0.5, vis_labels, 0.5, 0)

            vis_image = np.concatenate((f, labels_image), axis=1)

            ret = vid_out.write(vis_image)

        curr_pos = curr_pos + 1

    if vid_out:
        vid_out.release()

if __name__ == '__main__':
    args = parse_args()
    class_groups = get_class_groups()
    visualize(args.input_video_path,
              args.input_detections_path,
              args.output_video_path,
              args.scale_boxes,
              class_groups,
              max_frames=args.max_frames)
