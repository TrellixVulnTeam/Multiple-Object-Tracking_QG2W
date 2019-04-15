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
tfrecord_dir = '/n/pana/scratch/ravi/video_distillation_tfrecords_half/'
sequence_to_video_list = video_distillation.get_sequence_to_video_list(dataset_dir,
                                video_distillation.video_sequences_stable)

for s in video_distillation.video_sequences_stable:
    segment_dir = os.path.join(dataset_dir, s)
    segments = [ os.path.join(segment_dir, v[0]) for v in sequence_to_video_list[s] ]
    num_segments = 0
    for seg in segments:
        if num_segments > 0:
            break
        segment_name = seg.split('/')[-1].split('.')[0]
        detection_file = 'detectron_large_mask_rcnn_1_' + segment_name + '.npy'
        tfrecord_file = 'detectron_large_mask_rcnn_1_' + segment_name + '.tfrecords'
        detection_path = os.path.join(segment_dir, detection_file)
        tfrecord_path = os.path.join(tfrecord_dir, tfrecord_file)

        num_segments = num_segments + 1
        if os.path.exists(detection_path):
            print(seg, detection_path, tfrecord_path)
            if os.path.exists(tfrecord_path):
                continue
            mask_rcnn_tfrecords.segment_to_tfrecords(seg,
                    detection_path, tfrecord_path, 720,
                    1280, 5, max_frames=15000)
