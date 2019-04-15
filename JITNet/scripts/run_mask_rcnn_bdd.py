from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob

sys.path.append(os.path.realpath('./datasets'))
import video_distillation
import mask_rcnn_tfrecords

dataset_dir = '/n/pana/scratch/ravi/bdd/bdd100k/seg_videos/train'
output_dir = '/n/pana/scratch/ravi/mask_rcnn_bdd'

video_files = glob.glob(os.path.join(dataset_dir, '*.mp4'))
segment_names = [ v.split('/')[-1].split('.')[0] for v in video_files ]

for v, s in zip(video_files, segment_names):
    segment_out_name = 'detectron_large_mask_rcnn_1_' + s + '.npy'
    out_path = os.path.join(output_dir, segment_out_name)
    if os.path.exists(out_path):
        continue
    else:
        cmd_string = 'source scripts/run_detectron.sh %s %s $1'%(v, out_path)
        print(cmd_string)
