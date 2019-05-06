import os
from os.path import realpath, dirname, join
import sys
import argparse
import numpy as np
import glob
import cv2
import torch
import random
import json
from moviepy.editor import ImageSequenceClip

sys.path.append(os.path.realpath('./code'))
sys.path.append(os.path.realpath('../JITNet/utils'))

from net import SiamRPNvot
from stream import VideoInputStream
from SingleTracker import track_single_instance
import config
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect
from mask_rcnn_utils import bb_intersection_over_union

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate frames from video')
    parser.add_argument('--detections', required=True, help='numpy file containing detectron outputs')
    parser.add_argument('--video_path', required=True, help='input video')
    parser.add_argument('--output_folder', required=True, help='output folder path')
    parser.add_argument('--visualize', default=False, action='store_true', help='visualization flag')

    args = parser.parse_args()
    return args

def get_bbox_coords(frame_detections):
    boxes, _, scores, _ = frame_detections

    confident_bboxes = np.where((scores > config.conf_thresh) > 0)
    frame_boxes = []

    for idx in confident_bboxes[0]:
        x1, y1, x2, y2 = boxes[idx][1], boxes[idx][0], boxes[idx][3], boxes[idx][2]
        frame_boxes.append([int(x1 * config.flags_width), int(y1 * config.flags_height), int(x2 * config.flags_width), int(y2 * config.flags_height)])

    return frame_boxes

def check_iou_match(s_box, frame_boxes):
    best_iou = 0
    best_box_idx = None

    for idx, curr_box in enumerate(frame_boxes):
        curr_iou = bb_intersection_over_union(s_box, curr_box)

        if ((curr_iou >= best_iou)):
            best_box_idx = idx
            best_iou = curr_iou

    if (best_iou < config.bbox_match_thresh):
        best_box_idx = None

    return best_box_idx

def visualize_predictions(video_path, tracks, output_folder):
    images_folder = output_folder + "/final_tracks"
    os.mkdir(images_folder)

    s = VideoInputStream(video_path)
    frame_id = 0

    track_colors = [random.sample(range(0,255), 3) for i in range(len(tracks))]

    for im in s:
        assert im is not None
        for tid, track in enumerate(tracks):
            start_frame = track['start_frame']
            end_frame = track['end_frame']
            if ((frame_id >= start_frame) and (end_frame is None or frame_id < end_frame)):
                x1, y1, x2, y2 = track['track'][frame_id - start_frame]
                cv2.rectangle(im, (x1, y1), (x2, y2), track_colors[tid], 3)
                img_path = images_folder + "/frame_" + str(frame_id).zfill(len(str(s.length))) + "_vis.png"
                cv2.imwrite(img_path, im)
        print("Covered All Tracks for Frame: ", frame_id)
        frame_id += 1

    #############################################
     ## WRITE FRAME TRACKING OUTPUT TO VIDEO ##
    #############################################
    clip = ImageSequenceClip(images_folder, fps=29.97)
    clip.write_videofile(output_folder + "/final_tracks.mp4")
    return

def make_serializable(t):
    del t['active']
    del t['state']
    return t

def main():
    args = parse_args()
    detections = np.load(args.detections)[()]

    os.mkdir(args.output_folder)
    
    print("Loading net...")
    net = SiamRPNvot()
    net.load_state_dict(torch.load(join(realpath('./pretrained_models/'), 'SiamRPNOTB.model')))
    net.eval().cuda()

    stream = VideoInputStream(args.video_path)
    frame_id = 0

    data = {}
    data['tracks'] = []

    for im in stream:
        print("Frame ID: ", frame_id)
        assert im is not None
        # frame_boxes: array of [cx, cy, w, h] for each detection in frame
        frame_boxes = get_bbox_coords(detections[frame_id])
        matched_boxes = []

        # Match as many tracks as possible to existing detections
        for track in data['tracks']:

            if (not track['active']):
                # inactive track
                continue

            track['state'] = SiamRPN_track(track['state'], im)
            # TODO: Need to use states for IOU matches rather than previous detection
            box_id = check_iou_match(track['track'][-1], frame_boxes)
            
            if box_id is not None:
                # Matching detection box found
                track['track'].append(frame_boxes[box_id])
                
                matched_boxes.append(box_id)
            else:
                # No matching boxes found
                track['active'] = False
                track['end_frame'] = frame_id

        # Start new tracks for remaining unmatched detections
        for box_id in range(len(frame_boxes)):
            if box_id not in matched_boxes:
                x1, y1, x2, y2 = frame_boxes[box_id]
                new_pos = np.array([(x2+x1)/2, (y2+y1)/2]).astype(int)
                new_sz = np.array([(x2-x1+1), (y2-y1+1)]).astype(int)

                new_state = SiamRPN_init(im, new_pos, new_sz, net)
                data['tracks'].append({'active': True, 'state': new_state, 'track': [frame_boxes[box_id]], 'start_frame': frame_id, 'end_frame': None})

        frame_id += 1

    print("Output Phase")

    data['tracks'] = list(map(make_serializable, data['tracks']))

    print("Track 0: ", data['tracks'][0])

    with open(args.output_folder + '/final_predictions.txt', 'w') as outfile:  
        json.dump(data, outfile)

    if (args.visualize):
        visualize_predictions(args.video_path, data['tracks'], args.output_folder)
    return


if __name__=="__main__":
    main()