from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys,os
sys.path.append('./datasets')
import video_distillation

dataset_dir = '/n/scanner/ravi/video_distillation'
sequence_to_video_list = video_distillation.get_sequence_to_video_list(dataset_dir,
                            dataset_dir, video_distillation.video_sequences)

for v in sequence_to_video_list:
    stats_path = os.path.join(dataset_dir, v, 'class_stats.npy')
    if os.path.exists(stats_path):
        print('========')
        print(v)
        print('========')
        video_distillation.display_stats(stats_path)
