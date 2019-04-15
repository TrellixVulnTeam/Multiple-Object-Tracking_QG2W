from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import bdd_tfrecords as bdd

#bdd.convert_to_tfrecords('train', '/n/pana/scratch/ravi/bdd/bdd100k/seg', '/n/pana/scratch/ravi/bdd/bdd100k/seg')
#bdd.convert_to_tfrecords('val', '/n/pana/scratch/ravi/bdd/bdd100k/seg', '/n/pana/scratch/ravi/bdd/bdd100k/seg')

bdd.convert_to_tfrecords('train', '/n/pana/scratch/ravi/bdd/bdd100k/seg_usup', '/n/pana/scratch/ravi/bdd/bdd100k/seg_usup')
