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
    parser.add_argument('--track_path', required=True, help='file containing json track')
    parser.add_argument('--output_path', required=True, help='output video path')
    parser.add_argument('--fps', type=int, default=29.97, help='output folder path')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.track_path) as track_file:
        track = json.load(track_file)
    
    video_path = track['filename']
    s = VideoInputStream(video_path, start_frame=track['start'])
    track_length = track['end'] - track['start'] + 1

    width, height = track['w'], track['h']

    frames = []
    frame_id = 0

    for im in s:
        assert im is not None
        im = cv2.resize(im, (width, height))
        y1 = int(track['boxes'][frame_id][0]*height)
        x1 = int(track['boxes'][frame_id][1]*width)
        y2 = int(track['boxes'][frame_id][2]*height)
        x2 = int(track['boxes'][frame_id][3]*width)

        cv2.rectangle(im, (x1,y1), (x2,y2), (0,0,0), 4)
        frames.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
        frame_id += 1

        if (frame_id >= track_length):
            break
    
    clip = ImageSequenceClip(frames, fps=args.fps)
    clip.write_videofile(args.output_path)
    return

if __name__=="__main__":
    main()






