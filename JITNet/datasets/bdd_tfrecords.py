from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import numpy as np
import tensorflow as tf
from PIL import Image
from StringIO import StringIO
import glob

def num_examples_per_epoch(split='train'):
    if split == 'train':
        return 7000
    elif split == 'val':
        return 1000
    else:
        assert(0)

def num_classes():
    return 19

def _bytes_feature(value):
    """Wrapper for inserting bytes features into Example proto."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
    """Wrapper for inserting float features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
    """Wrapper for inserting int64 features into Example proto."""
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

id2trainid = { 0:255, 1:255, 2:255, 3:255, 4:255, 5:255, 6:255,
               7:0, 8:1, 9:255, 10:2, 11:4, 12:255, 13:255, 14:255,
               15:3, 16:255, 17:255, 18:255, 19:255, 20:5, 21:255, 22:255,
               23:255, 24:255, 25:6, 26:7, 27:255, 28:9, 29:8, 30:10,
               31:11, 32:12, 33:18, 34:15, 35:13, 36:255, 37:17, 38:255,
               39:16, 40:14}

id2color = [(128, 64, 128), (244, 35, 232), ( 70, 70, 70), (102,102,156), (190,153,153),
            (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
            ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
            (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32)]

def convert_to_tfrecords(split, data_path, out_path):
    assert(split in ['train', 'val'])
    img_list = glob.glob(os.path.join(data_path, 'images', split, '*.jpg'))

    tfrecord_file_name = os.path.join(out_path, 'bdd_{}.tfrecords'.format(split))
    writer = tf.python_io.TFRecordWriter(tfrecord_file_name)

    for im_file_name in img_list:
        print(im_file_name)
        im_data = tf.gfile.FastGFile(im_file_name, 'r').read()
        im_hash = im_file_name.split('/')[-1].split('.')[0]
        label_file_name = os.path.join(data_path, 'labels', split, im_hash + '_train_id.png')

        label_data = tf.gfile.FastGFile(label_file_name, 'r').read()

        im = np.asarray(Image.open(StringIO(im_data)))
        rows, cols, depth = im.shape[0], im.shape[1], im.shape[2]

        im_data = np.asarray(Image.open(StringIO(im_data))).tostring()
        label_data = np.asarray(Image.open(StringIO(label_data)))
        label_data = label_data.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
                                    'height': _int64_feature(rows),
                                    'width' : _int64_feature(cols),
                                    'depth' : _int64_feature(depth),
                                    'image' : _bytes_feature(im_data),
                                    'labels': _bytes_feature(label_data),
                                    'hash' : _bytes_feature(im_hash),
                                   }))

        writer.write(example.SerializeToString())
    writer.close()

def get_dataset(filename,
                buffer_size = 100,
                batch_size = 4,
                num_epochs = 50):

    dataset = tf.data.TFRecordDataset([filename])
    def parser(record):
        keys_to_features = {'height': tf.FixedLenFeature((), tf.int64),
                            'width' : tf.FixedLenFeature((), tf.int64),
                            'depth' : tf.FixedLenFeature((), tf.int64),
                            'image' : tf.FixedLenFeature((), tf.string),
                            'labels' : tf.FixedLenFeature((), tf.string),
                            'hash' : tf.FixedLenFeature((), tf.string),
                           }
        parsed = tf.parse_single_example(record, keys_to_features)

        image = tf.decode_raw(parsed['image'], tf.uint8)
        labels = tf.decode_raw(parsed['labels'], tf.uint8)

        #height = tf.cast(parsed['height'], tf.int32)
        #width = tf.cast(parsed['width'], tf.int32)
        #depth = tf.cast(parsed['depth'], tf.int32)

        image_shape = tf.stack([720, 1280, 3])
        labels_shape = tf.stack([720, 1280, 1])

        image = tf.reshape(image, image_shape)
        labels = tf.reshape(labels, labels_shape)

        return image, labels

    dataset = dataset.map(parser)
    dataset = dataset.shuffle(buffer_size = buffer_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    return iterator

if __name__ == '__main__':

    iterator = \
        get_dataset('/n/pana/scratch/ravi/bdd/bdd100k/seg/bdd_train.tfrecords')
    batch_image, batch_labels = iterator.get_next()

    with tf.Session() as sess:
        for i in range(500):
            start = time.time()
            image_vals, labels_vals = sess.run([batch_image, batch_labels])
            end = time.time()
            print(i, image_vals.shape, labels_vals.shape, end - start)
