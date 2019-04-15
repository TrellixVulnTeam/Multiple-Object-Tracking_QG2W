import os
import sys
import argparse
import numpy as np
import cv2
import random
from moviepy.editor import ImageSequenceClip

sys.path.append(os.path.realpath('./code'))
sys.path.append(os.path.realpath('../JITNet/utils'))
from stream import VideoInputStream
from SingleTracker import track_single_instance

flags_height = 720
flags_width = 1280

def parse_args():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate frames from video')
    parser.add_argument('--detections', required=True, help='numpy file containing detectron outputs')
    parser.add_argument('--video_path', required=True, help='input video')
    parser.add_argument('--output_folder', required=True, help='output folder path')

    args = parser.parse_args()
    return args

def get_initial_bbox_coords(detections_path):
    detections = np.load(detections_path)[()]
    boxes, _, scores, _ = detections[0]

    conf_thresh = 0.5
    confident_bboxes = np.where((scores > conf_thresh) > 0)
    first_boxes = []

    for idx in confident_bboxes[0]:
        x1, y1, x2, y2 = boxes[idx][1], boxes[idx][0], boxes[idx][3], boxes[idx][2]
        first_boxes.append([x1 * flags_width, y1 * flags_height, x2 * flags_width, y2 * flags_height])

    first_boxes = np.array(first_boxes)
    first_boxes = first_boxes.astype(int)

    print("********Initial Frame Boxes*********")
    print(first_boxes.shape)
    print(first_boxes)
    return first_boxes

def visualize_predictions(video_path, tracks, output_folder):
    os.mkdir(output_folder)

    s = VideoInputStream(video_path)
    frame_id = 0

    track_colors = [random.sample(range(0,255), 3) for i in range(len(tracks))]

    for im in s:
        assert im is not None
        track_id = 0
        for track in tracks:
            curr_box = track[frame_id]
            cv2.rectangle(im, (curr_box[0], curr_box[1]), (curr_box[0] + curr_box[2], curr_box[1] + curr_box[3]), track_colors[track_id], 3)
            img_path = output_folder + "/frame_" + str(frame_id).zfill(len(str(s.length))) + "_vis.png"
            cv2.imwrite(img_path, im)
            track_id += 1
        print("Covered All Tracks for Frame: ", frame_id)
        frame_id += 1
    return

def main():
    args = parse_args()
    
    # Get Initial Bounding Box Coords (For Frame Zero)
    first_boxes = get_initial_bbox_coords(args.detections)
    tracks = []

    #############################################
     ## SEQUENTIALLY TRACK MULTIPLE INSTANCES ##
    #############################################
    for i in range(len(first_boxes)):
        print("Tracking Instance: ", i)
        track_output_folder = args.output_folder + str(i)
        current_track = track_single_instance(args.video_path, int(first_boxes[i][0]), int(first_boxes[i][1]), int(first_boxes[i][2]), int(first_boxes[i][3]), track_output_folder)
        tracks.append(current_track)
    
    tracks = np.array(tracks)


    images_folder = args.output_folder + "final_tracks"
    visualize_predictions(args.video_path, tracks, images_folder)

    #############################################
     ## WRITE FRAME TRACKING OUTPUT TO VIDEO ##
    #############################################
    clip = ImageSequenceClip(images_folder, fps=29.97)
    clip.write_videofile(args.output_folder + "final_tracks.mp4")
    return

if __name__=="__main__":
    main()