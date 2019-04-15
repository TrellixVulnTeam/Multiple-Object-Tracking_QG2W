import os
from os.path import realpath, dirname, join
import sys
import argparse
import numpy as np
import glob
import cv2
import torch
import random
from moviepy.editor import ImageSequenceClip

sys.path.append(os.path.realpath('./code'))
sys.path.append(os.path.realpath('../JITNet/utils'))

from net import SiamRPNvot
from stream import VideoInputStream
from SingleTracker import track_single_instance
import config
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate frames from video')
    parser.add_argument('--detections', required=True, help='numpy file containing detectron outputs')
    parser.add_argument('--video_path', required=True, help='input video')
    parser.add_argument('--output_folder', required=True, help='output folder path')

    args = parser.parse_args()
    return args

def get_bbox_coords(frame_detections):
    boxes, _, scores, _ = frame_detections

    conf_thresh = 0.5
    confident_bboxes = np.where((scores > conf_thresh) > 0)
    frame_boxes = []

    for idx in confident_bboxes[0]:
        x1, y1, x2, y2 = boxes[idx][1], boxes[idx][0], boxes[idx][3], boxes[idx][2]
        cx, cy, w, h = (x1+x2)/2 , (y1+y2)/2, x2-x1, y2-y1
        frame_boxes.append([cx * config.flags_width, cy * config.flags_height, w * config.flags_width, h * config.flags_height])
        # frame_boxes.append([x1 * config.flags_width, y1 * config.flags_height, x2 * config.flags_width, y2 * config.flags_height])

    frame_boxes = np.array(frame_boxes)
    frame_boxes = frame_boxes.astype(int)

    # print("********Frame Boxes*********")
    # print(frame_boxes.shape)
    # print(frame_boxes)
    return frame_boxes

def check_iou_match(state, frame_boxes):
    raise NotImplementedError

def main():
    args = parse_args()
    detections = np.load(args.detections_path)[()]
    
    net = SiamRPNvot()
    net.load_state_dict(torch.load(join(realpath('./pretrained_models/'), 'SiamRPNOTB.model')))
    net.eval().cuda()

    stream = VideoInputStream(args.video_path)
    frame_id = 0

    # tracks: (active, array of track states)
    tracks = []

    for im in stream:
        assert im is not None
        frame_boxes = get_bbox_coords(detections[frame_id])
        matched_boxes = []

        # Match as many tracks as possible to existing detections
        for tid, track in enumerate(tracks):
            new_state = SiamRPN_track(track[-1], im)
            matching_box_exists, box_id = check_iou_match(new_state, frame_boxes)
            
            if matching_box_exists:
                assert box_id is not None

                matched_boxes.append(box_id)
                # modify size, pos attributes of new state to matching box for tighter detection
                # append new_state to tracks[tid][1]
            
            # Otherwise, matching box doesn't exist so shut this track down
        
        # Start new tracks for remaining unmatched detections
        for box_id in range(length(frame_boxes)):
            if box_id not in matched_boxes:
                # Create new state with box info using SiamRPN_init
                # Append (true, state) to tracks

        frame_id += 1
    
    tracks = np.array(tracks)



            









if __name__=="__main__":
    main()