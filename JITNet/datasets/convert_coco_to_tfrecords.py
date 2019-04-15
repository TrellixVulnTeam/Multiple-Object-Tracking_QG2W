from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import coco_tfrecords as coco

coco.convert_to_tfrecords("/home/ubuntu/cv-research/JITNet/data/coco/images/train2017", "/home/ubuntu/cv-research/JITNet/data/coco/annotations/train2017", "/home/ubuntu/cv-research/JITNet/data/coco/images/output_train2017", "train")
# bdd.convert_to_tfrecords('train', '/n/pana/scratch/ravi/bdd/bdd100k/seg_usup', '/n/pana/scratch/ravi/bdd/bdd100k/seg_usup')
