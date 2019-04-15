import sys, os
import csv
import cv2
import numpy as np
sys.path.append('./utils')
sys.path.append('./datasets')
import video_distillation

group_list = [['badminton1', 'squash1', 'table_tennis1',
                 'softball1', 'hockey1', 'soccer1', 'tennis1',
                 'tennis2', 'tennis3', 'volleyball1', 'volleyball3',
                 'ice_hockey1', 'kabaddi1', 'figure_skating1',
                 'drone2'],
              ['elephant1', 'birds2', 'giraffe1', 'dogs2', 'horses1'],
              ['ice_hockey_ego_1', 'basketball_ego1', 'ego_dodgeball1',
               'ego_soccer1', 'biking1'],
              ['streetcam1', 'streetcam2', 'jackson_hole1', 'jackson_hole2',
               'samui_murphys1', 'samui_walking_street1', 'toomer1',
               'southbeach1'],
              ['driving1', 'walking1']]
flat_list = [ v for g in group_list for v in g ]

flat_list = ['figure_skating1']

video_dir = '/n/pana/scratch/ravi/video_distillation_final'
suffix = '_stride_8_frame_30000_thresh_0.9_0.01'
#image_dir = '/n/pana/scratch/ravi/video_distillation_images'

image_dir = '/n/pana/scratch/ravi/figure_skating_frames'

for s in flat_list:
    video_path = os.path.join(video_dir, s + suffix + '.mp4')
    if s == 'kabaddi1':
        continue
    print(video_path)
    v = cv2.VideoCapture(video_path)
    for idx in range(15000, 16000, 8):
        v.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, f = v.read()
        if f is None:
            continue
        #two_pane = f[:, 1280:, :]
        two_pane = f[:, 1280*2:1280*3, :]
        img_name = os.path.join(image_dir, s + str(idx) + '.png')
        cv2.imwrite(img_name, two_pane)
        print(two_pane.shape)
