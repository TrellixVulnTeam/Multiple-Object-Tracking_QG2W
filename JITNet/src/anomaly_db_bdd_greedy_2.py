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
from moviepy.editor import ImageSequenceClip

sys.path.append(os.path.realpath('./src'))
sys.path.append(os.path.realpath('./datasets'))

import video_distillation

from mask_rcnn_tfrecords import get_dataset, batch_segmentation_masks,\
                                visualize_masks
from mask_rcnn_utils import mask_rcnn_get_best_attention_match,\
                            mask_rcnn_get_best_box_match,\
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

tf.app.flags.DEFINE_string('anomaly_out_dir',
        None, 'Path to output directory for anomalies.')

tf.app.flags.DEFINE_string('frames_out_dir',
        None, 'Path to output directory for individual frames.')

tf.app.flags.DEFINE_boolean(
    'scale_boxes', True,
    'Scale the boxes to image height and width.')

FLAGS = tf.app.flags.FLAGS

def filter_box_with_attention_map(attention_map, cls, boxes, masks, classes,
                                  scores, height, width, conf_thresh=0.5):

    high_conf_idx = scores > conf_thresh
    filtered_boxes = filterd_boxes[high_conf_idx]
    filtered_classes = filterd_classes[high_conf_idx]
    filtered_scores = filtered_scores[high_conf_idx]
    filtered_masks = filtered_masks[high_conf_idx]

    filtered_boxes[:, 0] = filtered_boxes[:, 0] * height
    filtered_boxes[:, 2] = filtered_boxes[:, 2] * height

    filtered_boxes[:, 1] = filtered_boxes[:, 1] * width
    filtered_boxes[:, 3] = filtered_boxes[:, 3] * width

    print(filtered_boxes.shape)

def get_objects_of_interest(boxes, scores, classes, masks, object_ids):
    interest_mask = np.zeros(classes.shape)
    for id in object_ids:
        interest_mask = np.logical_or(interest_mask, classes == id)
    return boxes[interest_mask], scores[interest_mask], \
           classes[interest_mask], masks[interest_mask]

def main(_):
    if FLAGS.anomaly_out_dir is not None and os.path.exists(FLAGS.anomaly_out_dir):
        print('Variant seems to have been already run')
        exit(0)

    if FLAGS.anomaly_out_dir is not None:
        os.mkdir(FLAGS.anomaly_out_dir)

    if FLAGS.frames_out_dir is not None and os.path.exists(FLAGS.frames_out_dir):
        print('Variant seems to have been already run')
        exit(0)

    if FLAGS.frames_out_dir is not None:
        os.mkdir(FLAGS.frames_out_dir)

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

    pick_threshold = 0.5
    area_threshold = 0.05 * FLAGS.height * 0.05 * FLAGS.width

    discrepancy = False
    input_streams = MaskRCNNSequenceStream(video_files, detections_paths,
                                           start_frame=0, stride=1)
    curr_frame = 0
    prev_frame = np.zeros((FLAGS.height, FLAGS.width, 3), dtype=np.uint8)

    for frame, boxes, classes, scores, \
        masks, num_objects, frame_id in input_streams:
        
        if (curr_frame == 0):
            print(boxes)
        
        
        if curr_frame > FLAGS.max_frames:
            break

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

            """
            picked_idx, picked_iou, picked_area = \
                mask_rcnn_get_best_attention_match(boxes, masks,
                                                   scores,
                                                   attn_mask,
                                                   (FLAGS.height, FLAGS.width),
                                                   match_threshold=match_threshold)
            """
            picked_idx, picked_iou, picked_area = \
                mask_rcnn_get_best_box_match(boxes, masks,
                                             scores,
                                             prev_bbox,
                                             (FLAGS.height, FLAGS.width),
                                             match_threshold=match_threshold)

            # End track if match not found in current frame
            if picked_idx < 0:
                print('Lost Track', tid)
                prev_area = (prev_bbox[2] - prev_bbox[0]) * \
                            (prev_bbox[3] - prev_bbox[1]) * \
                            FLAGS.height * FLAGS.width
                if picked_iou < miss_threshold and prev_iou >= match_threshold and \
                   prev_area > area_threshold:
                    print('Anomaly', tid)
                    anomaly_id = anomaly_id + 1

                    x1 = int(prev_bbox[0] * FLAGS.height)
                    y1 = int(prev_bbox[1] * FLAGS.width)
                    x2 = int(prev_bbox[2] * FLAGS.height)
                    y2 = int(prev_bbox[3] * FLAGS.width)

                    preds_image = frame[0].copy()
                    prev_vis = prev_frame.copy()
                    cv2.rectangle(prev_vis, (y1, x1), (y2, x2),
                                  (0, 0, 255), 4)

                    for bidx in range(len(boxes)):
                        if scores[bidx] > pick_threshold:
                            x1 = int(boxes[bidx][0] * FLAGS.height)
                            y1 = int(boxes[bidx][1] * FLAGS.width)
                            x2 = int(boxes[bidx][2] * FLAGS.height)
                            y2 = int(boxes[bidx][3] * FLAGS.width)
                            cv2.rectangle(preds_image, (y1, x1), (y2, x2),
                                          (0, 255, 0), 4)

                    vis_image = np.concatenate((preds_image, prev_vis), axis=0)
                    img_path = os.path.join(FLAGS.anomaly_out_dir,
                                            'anomaly_' + str(anomaly_id) + '_vis.png')
                    cv2.imwrite(img_path, vis_image)

                assert(tid not in tracks)
                tracks[tid] = active_tracks[tid]
                active_tracks.pop(tid)
            else:
                print('Tracking', tid)
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

        print(high_conf_idx[0])
        print(untracked_idx)
        print(tracked_detections[curr_frame])

        for uidx in untracked_idx:
            track_info = (curr_frame, boxes[uidx],
                          masks[uidx], classes[uidx],
                          scores[uidx], 0.0)
            tracked_detections[curr_frame].append(uidx)
            track_id = track_id + 1
            active_tracks[track_id] = []
            print('Starting New Track', track_id)
            active_tracks[track_id].append(track_info)

        curr_frame = curr_frame + 1
        prev_frame = frame[0]

    # End all active tracks
    for tid in active_tracks.keys():
        assert(tid not in tracks)
        tracks[tid] = active_tracks[tid]
        active_tracks.pop(tid)

    # video_out_path = os.path.join(FLAGS.anomaly_out_dir, 'tracks_vis.avi')
    visualize_tracks(FLAGS.video_path, tracks,
                     FLAGS.video_out_path,
                     FLAGS.height, FLAGS.width,
                     FLAGS.max_frames)

def visualize_tracks(video_in_path, tracks,
                     video_out_path,
                     height, width,
                     max_frames):

    frame_to_tracks = {}
    track_count = 0
    for t in tracks.keys():
        color = random.sample(range(0,255), 3)
        for f, box, _, _, _, _ in tracks[t]:
            if f not in frame_to_tracks:
                frame_to_tracks[f] = []
            frame_to_tracks[f].append((box, color, t))
        track_count += 1

    s = VideoInputStream(video_in_path)
    curr_frame = 0

    # vid_out = None
    # if video_out_path:
    #     rate = s.rate
    #     vid_out = cv2.VideoWriter(video_out_path,
    #                               cv2.VideoWriter_fourcc(*'X264'),
    #                               rate, (width, height))

    for f in s:
        if curr_frame > max_frames:
            break
        f = cv2.resize(f, (FLAGS.width, FLAGS.height))
        box_idx = 0
        if curr_frame in frame_to_tracks:
            for b, color, track in frame_to_tracks[curr_frame]:
                x1 = int(b[0] * height)
                y1 = int(b[1] * width)
                x2 = int(b[2] * height)
                y2 = int(b[3] * width)

                ################
                # SANITY CHECK #
                ################
                # if (curr_frame == 0 or curr_frame == 60):
                #     bounding_box = f[x1:x2, y1:y2]
                #     cv2.imwrite("/home/ubuntu/cv-research/pytorch-mobilenet-v2/frame_{}/track_{}.jpg".format(curr_frame, track), bounding_box)
                # if (curr_frame == 0):
                #     print("X1: %d, Y1: %d, X2: %d, Y2: %d" % (y1, x1, y2, x2))

                cv2.rectangle(f, (y1,x1), (y2,x2), color, 4)
                box_idx += 1

        # ret = vid_out.write(f)
        img_path = os.path.join(FLAGS.frames_out_dir, 'frame_' + str(curr_frame).zfill(len(str(s.length))) + '_vis.png')
        cv2.imwrite(img_path, f)

        curr_frame = curr_frame + 1

    clip = ImageSequenceClip(FLAGS.frames_out_dir, fps=29.97)
    clip.write_videofile("./Detectron.pytorch/video_inference/greedy_tracks.mp4")

    # vid_out.release()

if __name__ == '__main__':
  tf.app.run()
