from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import glob

sys.path.append(os.path.realpath('./datasets'))
import video_distillation
import mask_rcnn_tfrecords

dataset_dir = '/n/scanner/ravi/video_distillation'
output_dir = '/n/pana/scratch/ravi/video_distillation_small_mask_rcnn'

sequence_to_video_list = video_distillation.get_sequence_to_video_list(dataset_dir,
                                video_distillation.video_sequences_stable)

for s in video_distillation.video_sequences_stable:
    segment_dir = os.path.join(dataset_dir, s)
    segments = [ os.path.join(segment_dir, v[0]) for v in sequence_to_video_list[s] ]
    segment_out_dir = os.path.join(output_dir, s)
    if not os.path.exists(segment_out_dir):
        os.makedirs(segment_out_dir)
    for seg in segments:
        segment_name = seg.split('/')[-1].split('.')[0]
        out_name = 'detectron_small_mask_rcnn_1_' + segment_name + '.npy'
        out_path = os.path.join(output_dir, out_name)
        if os.path.exists(out_path):
            continue
        else:
            cmd_string = 'source scripts/run_detectron.sh %s %s $1'%(seg, out_path)
            print(cmd_string)
