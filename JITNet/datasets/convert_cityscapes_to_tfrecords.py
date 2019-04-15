from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cityscapes_tfrecords as cityscapes

#cityscapes.convert_to_tfrecords('train', '/data/ravi/cityscapes', '/data/ravi/cityscapes', encoded = False)
#cityscapes.convert_to_tfrecords('val', '/data/ravi/cityscapes', '/data/ravi/cityscapes', encoded = False)
#cityscapes.convert_to_tfrecords('test', '/data/ravi/cityscapes', '/data/ravi/cityscapes', encoded = False)

#cityscapes.convert_coarse_to_tfrecords('/n/scanner/datasets/cityscapes', '/n/scanner/datasets/cityscapes', encoded = False)
cityscapes.convert_xception_to_tfrecords('/n/scanner/datasets/cityscapes', '/n/pana/scratch/ravi/cityscapes/xception',
                                         '/n/pana/scratch/ravi/cityscapes/xception/compressed', encoded = True)

#cityscapes.convert_self_to_tfrecords('/n/pana/scratch/ravi/cityscapes/self',
#                                     '/n/pana/scratch/ravi/cityscapes/', encoded = False)
