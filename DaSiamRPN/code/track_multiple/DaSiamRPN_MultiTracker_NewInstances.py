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
    state_pos, state_sz = state['target_pos'], state['target_sz']
    s_cx, s_cy = state_pos[0], state_pos[1]
    s_w, s_h = state_sz[0], state_sz[1]
    s_x1, s_y1 = s_cx-s_w/2, s_cy-s_h/2
    s_box = [s_x1, s_y1, s_x1+s_w, s_y1+s_h]

    best_iou = 0
    best_box_idx = None

    for idx, box in enumerate(frame_boxes):
        [cx, cy, w, h] = box
        x2, y2 = cx-w/2, cy-h/2
        curr_box = [x2, y2, x2+w, y2+h]

        curr_iou = bb_intersection_over_union(s_box, curr_box)

        if ((curr_iou > best_iou) and (curr_iou > config.bbox_match_thresh)):
            best_box_idx = idx
            best_iou = curr_iou

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
            if (frame_id >= start_frame and (end_frame is None or frame_id < end_frame)):
                curr_box = track['track'][frame_id - start_frame]
                [cx, cy] = curr_box['target_pos']
                [w, h] = curr_box['target_sz']
                cv2.rectangle(im, (int(cx-w/2), int(cy-h/2)), (int(cx+w/2), int(cy+h/2)), track_colors[tid], 3)
                img_path = images_folder + "/frame_" + str(frame_id).zfill(len(str(s.length))) + "_vis.png"
                cv2.imwrite(img_path, im)
        print("Covered All Tracks for Frame: ", frame_id)
        frame_id += 1

    #############################################
     ## WRITE FRAME TRACKING OUTPUT TO VIDEO ##
    #############################################
    clip = ImageSequenceClip(images_folder, fps=29.97)
    clip.write_videofile(output_folder + "final_tracks.mp4")
    return

def make_track_serializable(state):
    new_state = {}
    new_state['target_pos'] = list(map(lambda x: float(x), state['target_pos']))
    new_state['target_sz'] = list(map(lambda x: float(x), state['target_sz']))
    return new_state

def make_serializable(elem):
    elem['track'] = list(map(make_track_serializable, elem['track']))
    return elem

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

    # data: {tracks: [{active:T/F, track:[], start_frame:int, end_frame:int}]}
    data = {}
    data['tracks'] = []

    for im in stream:
        print("Frame ID: ", frame_id)
        assert im is not None
        # frame_boxes: array of [cx, cy, w, h] for each detection in frame
        frame_boxes = get_bbox_coords(detections[frame_id])
        matched_boxes = []

        # Match as many tracks as possible to existing detections
        for tid, track in enumerate(data['tracks']):

            if (not track['active']):
                # inactive track
                continue

            prev_state = track['track'][-1]
            new_state = SiamRPN_track(prev_state, im)
            box_id = check_iou_match(new_state, frame_boxes)
            
            if box_id is not None:
                matched_boxes.append(box_id)
                new_box = frame_boxes[box_id]

                pos, sz = np.array(new_box[:2]), np.array(new_box[2:])
                new_state['target_pos'] = pos
                new_state['target_sz'] = sz

                data['tracks'][tid]['track'].append(new_state)
            else:
                # Otherwise, no matching boxes found
                data['tracks'][tid]['active'] = False
                data['tracks'][tid]['end_frame'] = frame_id
        
        # Start new tracks for remaining unmatched detections
        for box_id in range(len(frame_boxes)):
            if box_id not in matched_boxes:
                new_pos, new_sz = np.array(frame_boxes[box_id][:2]), np.array(frame_boxes[box_id][2:])
                new_state = SiamRPN_init(im, new_pos, new_sz, net)
                data['tracks'].append({'active': True, 'track': [new_state], 'start_frame': frame_id, 'end_frame': None})

        frame_id += 1

    print("Output Phase")

    data['tracks'] = list(map(make_serializable, data['tracks']))

    with open(args.output_folder + '/final_predictions.txt', 'w') as outfile:  
        json.dump(data, outfile)

    if (args.visualize):
        visualize_predictions(args.video_path, data['tracks'], args.output_folder)
    return


if __name__=="__main__":
    main()