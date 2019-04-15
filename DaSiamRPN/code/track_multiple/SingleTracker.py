# --------------------------------------------------------
# DaSiamRPN
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
#!/usr/bin/python

import glob, cv2, torch
import numpy as np
import sys
import os
from os.path import realpath, dirname, join

sys.path.append(os.path.realpath('./code'))
sys.path.append(os.path.realpath('../JITNet/utils'))

from net import SiamRPNvot
from stream import VideoInputStream
from run_SiamRPN import SiamRPN_init, SiamRPN_track
from utils import get_axis_aligned_bbox, cxy_wh_2_rect

def track_single_instance(video_path, x1, y1, x2, y2, output_folder):
    os.mkdir(output_folder)

    # load net
    net = SiamRPNvot()
    net.load_state_dict(torch.load(join(realpath('./pretrained_models/'), 'SiamRPNOTB.model')))
    net.eval().cuda()

    init_rbox = [x1, y1, x2, y1, x1, y2, x2, y2]
    [cx, cy, w, h] = get_axis_aligned_bbox(init_rbox)

    target_pos, target_sz = np.array([cx, cy]), np.array([w, h])
    
    s = VideoInputStream(video_path)
    frame_id = 0
    toc = 0
    state = None

    current_track = []
    res = [x1, y1, x2-x1, y2-y1]
    current_track.append(res)

    for im in s:
        assert im is not None

        if (frame_id == 0):
            state = SiamRPN_init(im, target_pos, target_sz, net)
            frame_id += 1
            continue
        
        assert state is not None

        # tracking and visualization
        tic = cv2.getTickCount()
        state = SiamRPN_track(state, im)  # track
        toc += cv2.getTickCount()-tic
        res = cxy_wh_2_rect(state['target_pos'], state['target_sz'])
        res = [int(l) for l in res]

        current_track.append(res)

        cv2.rectangle(im, (res[0], res[1]), (res[0] + res[2], res[1] + res[3]), (0, 255, 255), 3)
        img_path = output_folder + "/frame_" + str(frame_id) + "_vis.png"
        cv2.imwrite(img_path, im)
        frame_id += 1

    current_track = np.array(current_track)
    return current_track