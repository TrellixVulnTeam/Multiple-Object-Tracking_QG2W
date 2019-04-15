from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
sys.path.append('./datasets')
import video_distillation

dataset_dir = '/n/scanner/ravi/video_distillation'
sequence_to_video_list = video_distillation.get_sequence_to_video_list(dataset_dir,
                            video_distillation.video_sequences)

print(sequence_to_video_list)
video_distillation.get_dataset_stats(sequence_to_video_list, dataset_dir, 0.5,
                                     max_segments=2)
