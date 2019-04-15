from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os, sys
import time
import math
import tensorflow as tf
import matplotlib.pyplot as plt

from copy import deepcopy

import cv2

# import necessary pwcnet (tfoptflow) functions

sys.path.append(os.path.realpath('./tfoptflow/tfoptflow'))

from model_pwcnet import ModelPWCNet, _DEFAULT_PWCNET_TEST_OPTIONS
from visualize import plot_img_pairs_w_flows
from optflow import flow_to_img, flow_write_as_png

LG_CKPT_PATH = '../pwcnet/pwcnet_lg/pwcnet.ckpt-595000'
SM_CKPT_PATH = '../pwcnet/pwcnet_sm/pwcnet.ckpt-592000'

# create flags not present in online_scene_distillation
tf.app.flags.DEFINE_integer('start_frame', 0, 'Start frame')

# import functions from the online_scene_seg main file

# from online_scene_seg import get_class_groups, update_stats
from online_scene_distillation import update_stats

# Mask R-CNN utilities

sys.path.append(os.path.realpath('./datasets'))
sys.path.append(os.path.realpath('./utils'))

from mask_rcnn_tfrecords import get_dataset, batch_segmentation_masks,\
                                visualize_masks
from mask_rcnn_stream import MaskRCNNSequenceStream
import video_distillation
from video_distillation import sequence_to_class_groups_stable

from flow_cython import flow


def main():
    # initialize PWCNet in test mode

    FLAGS = tf.app.flags.FLAGS

    # set variables using flags
    max_frames = FLAGS.max_frames
    training_stride = FLAGS.training_stride
    stats_path = FLAGS.stats_path
    start_frame = FLAGS.start_frame
    
    height = FLAGS.height
    width = FLAGS.width
    
    nn_opts = deepcopy(_DEFAULT_PWCNET_TEST_OPTIONS)
    nn_opts['verbose'] = True
    nn_opts['ckpt_path'] = LG_CKPT_PATH
    nn_opts['batch_size'] = 1
    nn_opts['gpu_devices'] = ['/device:GPU:1']
    nn_opts['controller'] = '/device:GPU:1'
    nn_opts['use_dense_cx'] = True
    nn_opts['use_res_cx'] = True
    nn_opts['pyr_lvls'] = 6
    nn_opts['flow_pred_lvl'] = 2

    # since the model generates flow padded to multiples of 64
    # reduce back to the input video size
    nn_opts['adapt_info'] = (1, height, width, 2)
    
    nn = ModelPWCNet(mode='test', options=nn_opts)
    
    # assemble video files and detection paths using video_distillation
    video_files = []
    detections_paths = []
    
    sequence_to_video_list = \
            video_distillation.get_sequence_to_video_list(
                    FLAGS.dataset_dir,
                    video_distillation.video_sequences_stable)
    assert(FLAGS.sequence in sequence_to_video_list)
    sequence_path = os.path.join(FLAGS.dataset_dir, FLAGS.sequence)

    assert(os.path.isdir(sequence_path))
    assert(FLAGS.sequence in video_distillation.sequence_to_class_groups_stable)

    num_sequences = 0
    for s in sequence_to_video_list[FLAGS.sequence]:
        video_files.append(os.path.join(sequence_path, s[0]))
        detections_paths.append(os.path.join(sequence_path, s[1]))
        num_sequences = num_sequences + 1
        if num_sequences >= FLAGS.sequence_limit:
            break

    class_groups = \
            sequence_to_class_groups_stable[FLAGS.sequence]

    print(video_files)
    print(detections_paths)
    print(class_groups)

    class_groups = [ [video_distillation.detectron_classes.index(c) for c in g] \
                     for g in class_groups ]
    num_classes = len(class_groups) + 1
        

    input_streams = MaskRCNNSequenceStream(video_files, 
                                           detections_paths,
                                           start_frame=start_frame,
                                           stride=1)

    # initialize metrics

    curr_frame = 0
    prev_frame = None
    prev_in_frame = None
    pred = None
    per_frame_stats = {}
    num_classes = len(class_groups) + 1

    class_correct = np.zeros(num_classes, np.float32)
    class_total = np.zeros(num_classes, np.float32)
    class_tp = np.zeros(num_classes, np.float32)
    class_fp = np.zeros(num_classes, np.float32)
    class_fn = np.zeros(num_classes, np.float32)
    class_iou = np.zeros(num_classes, np.float32)

    pos_matrix = np.zeros(width * height)
    for i in range(width * height):
        pos_matrix[i] = i
    pos_matrix_idx = np.int32(pos_matrix)

    for frame, boxes, classes, scores, masks, num_objects, frame_id in input_streams:
        if curr_frame >= max_frames:
            break
        start = time.time()

        frame = cv2.resize(frame, (width, height))

        boxes = np.expand_dims(boxes, axis=0)
        classes = np.expand_dims(classes, axis=0)
        scores = np.expand_dims(scores, axis=0)
        masks = np.expand_dims(masks, axis=0)
        num_objects = np.expand_dims(num_objects, axis=0)

        labels_vals, _ = batch_segmentation_masks(1,
                                                  (height, width),
                                                  boxes, classes, masks, scores,
                                                  num_objects, True,
                                                  class_groups)
        labels_val = np.reshape(labels_vals, (height, width))
        if prev_frame is None:
            prev_frame = frame

        if curr_frame % training_stride == 0:
            pred = labels_val
            pred_ext = np.reshape(pred, (1, height, width))
            gt_pred = labels_val
            gt_im = frame
            # update stats with the parent prediction
            update_stats(labels_vals, pred_ext, class_tp, class_fp, class_fn,
                         class_total, class_correct, np.ones(labels_vals.shape, dtype=np.bool),
                         None, curr_frame, True, None, per_frame_stats)
        else:
            # compute forward flow
            pred_flows = nn.predict_from_img_pairs([(prev_frame, frame)], 
                                                   batch_size=1, 
                                                   verbose=False)
            # shape: (height, width, 2)
            pred_flow = np.round(pred_flows[0]).astype(np.int32)

            # Cython function
            pred = flow(pred, pred_flow)

            # update stats for flow prediction
            pred_ext = np.reshape(pred, (1, height, width))
            update_stats(labels_vals, pred_ext, class_tp, class_fp, class_fn,
                         class_total, class_correct, np.ones(labels_vals.shape, dtype=np.bool),
                         None, curr_frame, False, None, per_frame_stats)

        end = time.time()
        prev_frame = frame
        if curr_frame in per_frame_stats:
            print("Frame: {:07d} iou: {:1.03f} time: {:1.03f}".format(curr_frame, per_frame_stats[curr_frame]["iou"][1], end - start))
        curr_frame += 1

    if stats_path:
        np.save(stats_path, [per_frame_stats])


if __name__ == '__main__':
    main()

