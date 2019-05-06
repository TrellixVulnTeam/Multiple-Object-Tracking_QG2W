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

# parse_args: Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Generate frames from video')
    parser.add_argument('--detections', required=True, help='numpy file containing detectron outputs')
    parser.add_argument('--video_path', required=True, help='input video')
    parser.add_argument('--output_folder', required=True, help='output folder path')
    parser.add_argument('--visualize', default=False, action='store_true', help='visualization flag')

    args = parser.parse_args()
    return args

# get_bbox_coords: Filters frame for high confidence detections and returns in
# array of [x1,y1,x2,y2]'s
def get_bbox_coords(frame_detections):
    boxes, _, scores, _ = frame_detections

    confident_bboxes = np.where((scores > config.conf_thresh) > 0)
    frame_boxes = []

    for idx in confident_bboxes[0]:
        x1, y1, x2, y2 = boxes[idx][1], boxes[idx][0], boxes[idx][3], boxes[idx][2]
        frame_boxes.append([int(x1 * config.flags_width), int(y1 * config.flags_height), int(x2 * config.flags_width), int(y2 * config.flags_height)])

    return frame_boxes

def visualize_predictions(video_path, tracks, output_folder):
    images_folder = output_folder + "/final_tracks"
    os.mkdir(images_folder)

    s = VideoInputStream(video_path)
    frame_id = 0

    track_colors = [random.sample(range(0,255), 3) for i in range(len(tracks))]

    for im in s:
        assert im is not None
        for tid, track in enumerate(tracks):
            x1, y1, w, h = track['frame_boxes'][frame_id]
            cv2.rectangle(im, (x1, y1), (x1+w, y1+h), track_colors[tid], 3)
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

    print("Loading Net...")
    net = SiamRPNvot()
    net.load_state_dict(torch.load(join(realpath('./pretrained_models/'), 'SiamRPNOTB.model')))
    net.eval().cuda()

    stream = VideoInputStream(args.video_path)
    frame_id = 0

    data = {}
    data['tracks'] = []

    for im in stream:
        print("Tracking on Frame ID: ", frame_id)
        assert im is not None
        if frame_id == 0:
            frame_boxes = get_bbox_coords(detections[frame_id])
            
            #**************************
            # BAG DEBUGGING REMOVE!!!!!!
            #**************************
            frame_boxes.append([0,0,0,0])

            for box in frame_boxes:
                print("Box: ", box)
                x1, y1, x2, y2 = box

                # init_rbox = [x1, y1, x2, y1, x1, y2, x2, y2]
                init_rbox = [334.02,128.36,438.19,188.78,396.39,260.83,292.23,200.41]
                print("Init Box: ", init_rbox)
                cx, cy, w, h = get_axis_aligned_bbox(init_rbox)
                # cx, cy = float(x1+x2)/2, float(y1+y2)/2
                # w, h = float(x2-x1), float(y2-y1)


                pos, sz = np.array([cx, cy]), np.array([w, h])
                state = SiamRPN_init(im, pos, sz, net)

                curr_box = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                curr_box = [int(l) for l in curr_box]

                track = {'active': True, 'state': state, 'frame_boxes': [curr_box]}
                
                #**************************
                # BAG DEBUGGING REMOVE!!!!!!
                #**************************
                # track = {'active': True, 'state': state, 'frame_boxes': [[x1,y1,x2-x1,y2-y1]]}
                data['tracks'].append(track)
        else:
            for track in data['tracks']:
                state = SiamRPN_track(track['state'], im)
                curr_box = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
                curr_box = [int(l) for l in curr_box]

                track['state'] = state
                track['frame_boxes'].append(curr_box)
        
        frame_id += 1

    print("===========OUTPUT + VISUALIZATION===========")
    data['tracks'] = list(map(make_serializable, data['tracks']))
    print("Track 0: ", data['tracks'][0])

    if (args.visualize):
        visualize_predictions(args.video_path, data['tracks'], args.output_folder)


    return

if __name__=="__main__":
    main()