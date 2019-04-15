from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./datasets')
import video_distillation

dataset_dir = '/n/scanner/ravi/video_distillation'
vis_dir = '/n/pana/scratch/ravi/video_distillation_detectron'
sequence_to_video_list = video_distillation.get_sequence_to_video_list(dataset_dir,
                            dataset_dir, video_distillation.video_sequences)

video_distillation.visualize_detectron(sequence_to_video_list,
                            video_distillation.sequence_to_class_groups,
                            1, 15000, dataset_dir, vis_dir)
