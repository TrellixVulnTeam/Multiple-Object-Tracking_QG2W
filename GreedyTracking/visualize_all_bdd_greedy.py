import sys
import os
import glob

import numpy as np
import tensorflow as tf
import time
import cv2
import random
import json
import argparse
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



def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate frames from video')
    parser.add_argument('--tracks_dir', required=True, help='folder containing all tracks')
    parser.add_argument('--output_path', required=True, help='output video path')
    parser.add_argument('--fps', type=int, default=29.97, help='output folder path')

    args = parser.parse_args()
    return args

def list_files(dir):
    r = []
    for root, _, files in os.walk(dir):
        for name in files:
            r.append(os.path.join(root, name))
    return r

def collect_tracks(r):
    tracks = []
    for track_filename in r:
        with open(track_filename) as track_file:
            track = json.load(track_file)
            tracks.append(track)
    return tracks

def visualize_tracks(tracks, args):
    video_path = tracks[0]['filename']
    width, height = tracks[0]['w'], tracks[0]['h']

    s = VideoInputStream(video_path)
    frame_id = 0
    frames = []

    track_colors = [random.sample(range(0,255), 3) for i in range(len(tracks))]

    for im in s:
        assert im is not None
        im = cv2.resize(im, (width, height))

        for tid, track in enumerate(tracks):
            if (track['start'] <= frame_id and frame_id <= track['end']):
                box_id = frame_id - track['start']
                box = track['boxes'][box_id]
                y1 = int(box[0]*height)
                x1 = int(box[1]*width)
                y2 = int(box[2]*height)
                x2 = int(box[3]*width)
                cv2.rectangle(im, (x1,y1), (x2,y2), track_colors[tid], 2)
                # cv2.rectangle(im, (int(curr_box[0]), int(curr_box[1])), (int(curr_box[0] + curr_box[2]), int(curr_box[1] + curr_box[3])), track_colors[track_id], 3)
        
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        if frame_id % 500 == 0:
            print("Covered All Tracks for Frame: ", frame_id)
        frame_id += 1

    clip = ImageSequenceClip(frames, fps=args.fps)
    clip.write_videofile(args.output_path)
    return

def main():
    args = parse_args()

    print("Listing Files...")
    r = list_files(args.tracks_dir)
    print("Num Files: ", len(r))
    print("Collecting Tracks...")
    tracks = collect_tracks(r)
    print("Num Tracks: ", len(tracks))
    # print("Track 0: ", tracks[0])
    print("Visualizing Tracks...")
    visualize_tracks(tracks, args)
    return

if __name__=="__main__":
    main()






