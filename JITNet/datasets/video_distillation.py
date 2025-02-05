from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys
import glob
import numpy as np

sys.path.append('./src')
from visualize_detections import visualize

video_sequences = ['hockey1', 'tennis1', 'tennis2', 'tennis3', \
                   'ice_hockey_ego_1', 'basketball_ego1', 'volleyball1', \
                   'volleyball2', 'volleyball3', 'ego_soccer1', 'soccer1', \
                   'dodgeball1', 'ego_dodgeball1', 'driving1', 'walking1', \
                   'jackson_hole1', 'toomer1', 'jackson_hole2', 'park1', \
                   'samui_walking_street1', 'cooking1', 'badminton1', \
                   'squash1', 'dogs1', 'drone1', 'biking1', 'giraffe1', \
                   'drone2', 'birds1', 'birds2', 'horses1', \
                   'samui_murphys1', 'dogs2', 'elephant1']

video_sequences_stable = ['hockey1', 'tennis1', 'tennis2', 'tennis3', \
                          'ice_hockey_ego_1', 'basketball_ego1', 'volleyball1', \
                          'volleyball3', 'ego_soccer1', 'soccer1', \
                          'ego_dodgeball1', 'driving1', 'walking1', \
                          'jackson_hole1', 'toomer1', 'jackson_hole2', \
                          'samui_walking_street1', 'badminton1', \
                          'squash1', 'biking1', 'giraffe1', \
                          'drone2', 'birds2', 'horses1', \
                          'samui_murphys1', 'dogs2', 'elephant1', \
                          'table_tennis1', 'ice_hockey1', \
                          'southbeach1', 'softball1', 'streetcam1', \
                          'streetcam2', 'figure_skating1', 'kabaddi1' ]

detectron_classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse',
    'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
    'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
    'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass',
    'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake',
    'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

sequence_to_class_groups_stable = {
        'ego_dodgeball1' : [ ['person'] ],
        'volleyball3' : [ ['person'] ],
        'volleyball1' : [ ['person'] ],
        'squash1' : [ ['person'] ],
        'driving1' : [ ['person'], [ 'car', 'bus', 'truck'],
                       [ 'motorcycle', 'bicycle'] ],
        'ego_soccer1' : [ ['person'] ],
        'basketball_ego1' : [ ['person'], ['car'] ],
        'jackson_hole1' : [ ['person'], [ 'car', 'bus', 'truck']],
        'jackson_hole2' : [ ['person'], [ 'car', 'bus', 'truck']],
        'badminton1' : [ ['person'] ],
        'soccer1' : [ ['person'] ],
        'giraffe1' : [ ['person'], ['giraffe'] ],
        'toomer1' : [ ['person'], [ 'car', 'bus', 'truck']],
        'drone1' : [ ['person'], [ 'car', 'bus', 'truck'],
                     [ 'motorcycle', 'bicycle'] ],
        'drone2' : [ ['person'] ],
        'biking1' : [ ['person'], ['motorcycle', 'bicycle'] ],
        'samui_walking_street1' : [ ['person'], [ 'car', 'bus', 'truck'],
                                    [ 'motorcycle', 'bicycle'] ],
        'horses1' : [ ['person'], ['horse'] ],
        'hockey1' : [ ['person'] ],
        'tennis1' : [ ['person'] ],
        'tennis2' : [ ['person'] ],
        'tennis3' : [ ['person'] ],
        'ice_hockey_ego_1' : [ ['person'] ],
        'walking1' : [ ['person'], [ 'car', 'bus', 'truck'],
                       [ 'motorcycle', 'bicycle'] ],
        'birds2' : [ ['bird'] ],
        'elephant1' : [ ['elephant'] ],
        'samui_murphys1' : [ ['person'], [ 'car', 'bus', 'truck'],
                                    [ 'motorcycle', 'bicycle'] ],
        'dogs2' : [ ['person'], ['dog'], [ 'car', 'bus', 'truck']],
        'table_tennis1' : [ ['person'] ],
        'biking2' : [ ['person'], [ 'car', 'bus', 'truck'],
                       [ 'motorcycle', 'bicycle'] ],
        'ice_hockey1' : [ ['person'] ],
        'southbeach1' : [ ['person'], [ 'car', 'bus', 'truck'],
                       [ 'motorcycle', 'bicycle'] ],
        'softball1' : [ ['person'] ],
        'streetcam1' : [ ['person'], [ 'car', 'bus', 'truck']],
        'streetcam2' : [ ['person'], [ 'car', 'bus', 'truck']],
        'figure_skating1' : [ ['person'] ],
        'kabaddi1' : [ ['person'] ],
}

sequence_to_class_groups = {
        'ego_dodgeball1' : [ ['person'],['sports ball']],
        'volleyball3' : [ ['person'] ],
        'volleyball2' : [ ['person'] ],
        'volleyball1' : [ ['person'] ],
        'squash1' : [ ['person'], ['tennis racket'] ],
        'driving1' : [ ['person'], [ 'car', 'bus', 'truck'],
                       [ 'motorcycle', 'bicycle'] ],
        'ego_soccer1' : [ ['person'] ],
        'basketball_ego1' : [ ['person'], ['car'], ['sports ball'] ],
        'jackson_hole1' : [ ['person'], [ 'car', 'bus', 'truck'],
                            [ 'motorcycle', 'bicycle'] ],
        'jackson_hole2' : [ ['person'], [ 'car', 'bus', 'truck'],
                            [ 'motorcycle', 'bicycle'] ],
        'badminton1' : [ ['person'], ['tennis racket'] ],
        'soccer1' : [ ['person'], ['sports ball'] ],
        'giraffe1' : [ ['person'], ['giraffe'] ],
        'toomer1' : [ ['person'], [ 'car', 'bus', 'truck'],
                      [ 'motorcycle', 'bicycle'] ],
        'drone1' : [ ['person'], [ 'car', 'bus', 'truck'],
                     [ 'motorcycle', 'bicycle'] ],
        'drone2' : [ ['person'] ],
        'biking1' : [ ['person'], ['motorcycle', 'bicycle'] ],
        'dodgeball1' : [ ['person'], ['sports ball'] ],
        'samui_walking_street1' : [ ['person'], [ 'car', 'bus', 'truck'],
                                    [ 'motorcycle', 'bicycle'] ],
        'horses1' : [ ['person'], ['horse'] ],
        'park1' : [ ['person'], ['umbrella'] ],
        'hockey1' : [ ['person'], ['sports ball'] ],
        'tennis1' : [ ['person'], ['sports ball'], ['tennis racket'] ],
        'tennis2' : [ ['person'], ['sports ball'], ['tennis racket'] ],
        'tennis3' : [ ['person'], ['sports ball'], ['tennis racket'] ],
        'ice_hockey_ego_1' : [ ['person'] ],
        'walking1' : [ ['person'], [ 'car', 'bus', 'truck'],
                       [ 'motorcycle', 'bicycle'] ],
        'birds1' : [ ['bird'] ],
        'birds2' : [ ['bird'] ],
        'dogs1' : [ ['dog'] ],
        'elephant1' : [ ['elephant'] ],
        'samui_murphys1' : [ ['person'], [ 'car', 'bus', 'truck'],
                                    [ 'motorcycle', 'bicycle'] ],
        'dogs2' : [ ['person'], ['dog'], [ 'car', 'bus', 'truck']],
}

def get_sequence_to_video_list(video_path, detections_path, sequence_list,\
                               prefix = 'detectron_large_mask_rcnn_1_'):
    sequence_to_video_list = {}
    for v in sequence_list:
        if os.path.isdir(os.path.join(video_path, v)) and \
                os.path.isdir(os.path.join(detections_path, v)):
            sequence_to_video_list[v] = []
            video_segs = glob.glob(os.path.join(video_path, v, '*.mp4'))
            video_seg_names = sorted([ s.split('/')[-1] for s in video_segs])

            dets = glob.glob(os.path.join(detections_path, v, '*detectron*npy'))
            det_names = [ d.split('/')[-1] for d in dets ]

            for sn in video_seg_names:
                det_name = prefix + sn.split('.')[0] + '.npy'
                if det_name in det_names:
                    sequence_to_video_list[v].append((sn, det_name))
    return sequence_to_video_list

def visualize_detectron(sequence_to_video_list, sequence_to_class_groups,
                        num_sequences, num_frames, dataset_path,
                        visualization_path):

    for v in sequence_to_video_list:
        sequence_path = os.path.join(dataset_path, v)
        if os.path.isdir(sequence_path):
            segment_list = sequence_to_video_list[v]
            if v not in sequence_to_class_groups:
                continue
            class_groups = sequence_to_class_groups[v]
            class_groups = [ [detectron_classes.index(c) for c in g] for g in class_groups ]
            for s in range(min(num_sequences, len(segment_list))):
                seg_name, det_name = segment_list[s]
                seg_path = os.path.join(sequence_path, seg_name)
                det_path = os.path.join(sequence_path, det_name)
                out_path = os.path.join(visualization_path,
                                        'visualize_' +
                                        seg_name.split('.')[0] + '.avi' )
                visualize(seg_path, det_path, out_path, True,
                          class_groups, max_frames=num_frames)

def get_dataset_stats(sequence_to_video_list, dataset_path,
                      score_threshold, max_segments=3):
    class_stats = {}
    for v in sequence_to_video_list:
        sequence_path = os.path.join(dataset_path, v)
        if os.path.isdir(sequence_path):
            save_path = os.path.join(sequence_path, 'class_stats.npy')
            print(v)
            if os.path.exists(save_path):
                continue
            segment_list = sequence_to_video_list[v]
            num_detections = np.zeros(len(detectron_classes), np.int64)
            for s in range(min(len(segment_list), max_segments)):
                _, det_name = segment_list[s]
                det_path = os.path.join(sequence_path, det_name)
                detections = np.load(det_path)[()]
                for f in sorted(detections.keys()):
                    boxes, classes, scores, masks = detections[f]
                    for s in range(len(scores)):
                        if scores[s] > score_threshold:
                            num_detections[classes[s]] = num_detections[classes[s]] + 1
            np.save(save_path, num_detections)
            print(num_detections)

def display_stats(stats_path):
    class_stats = np.load(stats_path)
    for c in range(class_stats.shape[0]):
        print(detectron_classes[c], class_stats[c])
