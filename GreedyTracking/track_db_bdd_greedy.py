from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob

import numpy as np
import tensorflow as tf
import time
import cv2
import random
import json
from moviepy.editor import ImageSequenceClip

sys.path.append(os.path.realpath('../JITNet/utils'))
sys.path.append(os.path.realpath('../JITNet/src'))
sys.path.append(os.path.realpath('../JITNet/datasets'))

import video_distillation

from mask_rcnn_tfrecords import get_dataset, batch_segmentation_masks,\
                                visualize_masks
from mask_rcnn_utils import mask_rcnn_get_best_box_match,\
                            get_full_resolution_mask
from mask_rcnn_stream import MaskRCNNMultiStream, MaskRCNNSequenceStream
from stream import VideoInputStream

#######################
# Dataset Flags #
#######################

tf.app.flags.DEFINE_string(
    'video_path', None, 'Directory containing the dataset.')

tf.app.flags.DEFINE_integer(
    'height', 720, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer(
    'width', 1280, 'The number of samples in each batch.')

tf.app.flags.DEFINE_integer('max_frames', 100000,
                            'The maximum number of training steps.')

tf.app.flags.DEFINE_string(
    'detections_prefix', None, 'Path to video file.')

tf.app.flags.DEFINE_string(
    'detections_path', '',
    'Path to the detections file')

tf.app.flags.DEFINE_string('video_out_path',
        None, 'Path to output video file.')
tf.app.flags.DEFINE_string('tracks_out_dir',
        None, 'Path to output directory to store tracks.')

tf.app.flags.DEFINE_boolean(
    'scale_boxes', True,
    'Scale the boxes to image height and width.')

FLAGS = tf.app.flags.FLAGS

def get_objects_of_interest(boxes, scores, classes, masks, object_ids):
    interest_mask = np.zeros(classes.shape)
    for id in object_ids:
        interest_mask = np.logical_or(interest_mask, classes == id)
    return boxes[interest_mask], scores[interest_mask], \
           classes[interest_mask], masks[interest_mask]

def main(_):
    if FLAGS.tracks_out_dir is not None and os.path.exists(FLAGS.tracks_out_dir):
        print('Variant seems to have been already run')
        exit(0)

    if FLAGS.tracks_out_dir is not None:
        os.mkdir(FLAGS.tracks_out_dir)

    assert(os.path.isfile(FLAGS.video_path))
    assert(os.path.isfile(FLAGS.detections_path))

    video_files = [ FLAGS.video_path ]
    detections_paths = [ FLAGS.detections_path ]

    print(video_files, detections_paths)

    class_groups = [ ['person'], [ 'car', 'bus', 'truck'],
                       [ 'motorcycle', 'bicycle'] ]
    object_ids = [ video_distillation.detectron_classes.index(c) \
                   for g in class_groups for c in g ]


    tracked_detections = {}
    tracks = {}
    active_tracks = {}
    track_id = 0
    anomaly_id = 0

    pick_threshold = 0.92
    area_threshold = 0.05 * FLAGS.height * 0.05 * FLAGS.width

    discrepancy = False
    input_streams = MaskRCNNSequenceStream(video_files, detections_paths,
                                           start_frame=0, stride=1)
    curr_frame = 0
    prev_frame = np.zeros((FLAGS.height, FLAGS.width, 3), dtype=np.uint8)

    print("Tracking...")
    for frame, boxes, classes, scores, masks, num_objects, \
                                    frame_id in input_streams:
        if curr_frame > FLAGS.max_frames:
            break

        if curr_frame % 500 == 0:
            print("Reached Frame: ", curr_frame)

        if curr_frame not in tracked_detections:
            tracked_detections[curr_frame] = []

        frame = cv2.resize(frame, (FLAGS.width, FLAGS.height))

        frame = np.expand_dims(frame, axis=0)
        boxes = np.expand_dims(boxes, axis=0)
        classes = np.expand_dims(classes, axis=0)
        scores = np.expand_dims(scores, axis=0)
        masks = np.expand_dims(masks, axis=0)
        num_objects = np.expand_dims(num_objects, axis=0)

        boxes, scores, classes, masks = \
                get_objects_of_interest(boxes, scores, classes,
                                        masks, object_ids)

        # For all the active tracks find matches in the current frame
        for tid in active_tracks.keys():
            _, prev_bbox, prev_mask, prev_cls,\
                    prev_score, prev_iou = active_tracks[tid][-1]

            match_threshold = 0.5
            miss_threshold = 0.1

            attn_mask = get_full_resolution_mask(prev_bbox,
                                                 prev_mask,
                                                 (FLAGS.height, FLAGS.width),
                                                 scale_boxes=True)

            picked_idx, picked_iou, picked_area = \
                mask_rcnn_get_best_box_match(boxes, masks,
                                             scores,
                                             prev_bbox,
                                             (FLAGS.height, FLAGS.width),
                                             match_threshold=match_threshold)

            # Anomaly: End track if match not found in current frame
            if picked_idx < 0:
                # print('Lost Track', tid)
                assert(tid not in tracks)
                tracks[tid] = active_tracks[tid]
                active_tracks.pop(tid)
            else:
                # print('Tracking', tid)
                # Track collision ??
                if picked_idx in tracked_detections[curr_frame]:
                    assert(tid not in tracks)
                    tracks[tid] = active_tracks[tid]
                    active_tracks.pop(tid)
                else:
                    tracked_detections[curr_frame].append(picked_idx)
                    active_tracks[tid].append((curr_frame,
                                               boxes[picked_idx],
                                               masks[picked_idx],
                                               classes[picked_idx],
                                               scores[picked_idx],
                                               picked_iou))

        # Start new tracks for all untracked detections in the current frame
        high_conf_idx = np.where((scores > pick_threshold) > 0)

        # Only start tracks for objects which are currently untracked
        untracked_idx = []
        for idx in high_conf_idx[0]:
            if idx not in tracked_detections[curr_frame]:
                untracked_idx.append(idx)

        for uidx in untracked_idx:
            track_info = (curr_frame, boxes[uidx],
                          masks[uidx], classes[uidx],
                          scores[uidx], 0.0)
            tracked_detections[curr_frame].append(uidx)
            track_id = track_id + 1
            active_tracks[track_id] = []
            # print('Starting New Track', track_id)
            active_tracks[track_id].append(track_info)

        curr_frame = curr_frame + 1
        prev_frame = frame[0]

    # End all active tracks
    for tid in active_tracks.keys():
        assert(tid not in tracks)
        tracks[tid] = active_tracks[tid]
        active_tracks.pop(tid)

    store_tracks(FLAGS.tracks_out_dir, FLAGS.video_path, tracks)

def store_tracks(tracks_out_path, video_path, tracks):
    for tid in tracks:
        track = tracks[tid]
        start_frame = track[0][0]
        end_frame = track[-1][0]
        boxes = map(lambda x: (x[1]).tolist(), track)
        curr_track = {'filename': video_path, 'start': start_frame, \
                            'end': end_frame, 'boxes': boxes,  \
                            'w': FLAGS.width, 'h': FLAGS.height}
        with open(tracks_out_path + '/track_' + str(tid) + '.txt', 'w') as outfile:  
            json.dump(curr_track, outfile)

if __name__ == '__main__':
  tf.app.run()
