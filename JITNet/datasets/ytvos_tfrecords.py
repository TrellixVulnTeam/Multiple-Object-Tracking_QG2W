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
import random

import input_preprocess

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

def convert_to_tfrecords(dataset_path, out_path, encoded=True):
    label_list = glob.glob(os.path.join(dataset_path, 'train/Annotations/*/*.png'))
    random.seed(0)
    random.shuffle(label_list)

    total = len(label_list)
    count = 0

    tfrecord_file_name = os.path.join(out_path, 'ytvos.tfrecords')
    writer = tf.python_io.TFRecordWriter(tfrecord_file_name)

    for label_path in label_list:
        frame_id = label_path.split('/')[-1].split('.')[0]
        seg_id = label_path.split('/')[-2]
        image_path = os.path.join(dataset_path, 'train_all_frames',
                                  'JPEGImages', seg_id, frame_id + '.jpg')
        assert(os.path.exists(image_path))

        print(label_path)
        print(count, total)

        label_data = tf.gfile.FastGFile(label_path, 'r').read()
        im_data = tf.gfile.FastGFile(image_path, 'r').read()
        im = np.asarray(Image.open(StringIO(im_data)))

        if len(im.shape) < 3:
            continue

        rows, cols, depth = im.shape[0], im.shape[1], im.shape[2]
        if rows != 720 or cols != 1280:
            continue

        if not encoded:
            im_data = np.asarray(Image.open(StringIO(im_data))).tostring()

            label_data = np.asarray(Image.open(StringIO(label_data)))
            label_data = label_data.tostring()
        else:
            label_data = np.asarray(Image.open(StringIO(label_data)))
            label_out = StringIO()
            Image.fromarray(label_data).save(label_out, 'png')
            label_data = label_out.getvalue()

        example = tf.train.Example(features=tf.train.Features(feature={
                                    'height': _int64_feature(rows),
                                    'width' : _int64_feature(cols),
                                    'depth' : _int64_feature(depth),
                                    'image' : _bytes_feature(im_data),
                                    'labels': _bytes_feature(label_data),
                                    'id' : _bytes_feature(image_path),
                                   }))

        writer.write(example.SerializeToString())
        count = count + 1

    writer.close()

def get_dataset(filename,
                buffer_size = 100,
                batch_size = 4,
                num_epochs = 50,
                num_parallel_calls = 8,
                encoded = True,
                data_augment = False):

    dataset = tf.data.TFRecordDataset([filename])
    def parser(record):
        keys_to_features = {'height': tf.FixedLenFeature((), tf.int64),
                            'width' : tf.FixedLenFeature((), tf.int64),
                            'depth' : tf.FixedLenFeature((), tf.int64),
                            'image' : tf.FixedLenFeature((), tf.string),
                            'labels' : tf.FixedLenFeature((), tf.string),
                            'id' : tf.FixedLenFeature((), tf.string),
                           }
        parsed = tf.parse_single_example(record, keys_to_features)

        if encoded:
            image = tf.image.decode_image(parsed['image'])
            labels = tf.image.decode_image(parsed['labels'])
        else:
            image = tf.decode_raw(parsed['image'], tf.uint8)
            labels = tf.decode_raw(parsed['labels'], tf.uint8)

        height = tf.cast(parsed['height'], tf.int32)
        width = tf.cast(parsed['width'], tf.int32)
        #depth = tf.cast(parsed['depth'], tf.int32)

        image_shape = tf.stack([height, width, 3])
        labels_shape = tf.stack([height, width, 1])

        image = tf.reshape(image, image_shape)
        labels = tf.reshape(labels, labels_shape)

        _, image, labels = \
                input_preprocess.preprocess_image_and_label(image, labels,
                                                            720, 1280,
                                                            min_scale_factor=1.0,
                                                            max_scale_factor=1.0,
                                                            scale_factor_step_size=0,
                                                            is_training=data_augment)

        return image, labels

    dataset = dataset.map(parser, num_parallel_calls=num_parallel_calls)
    dataset = dataset.shuffle(buffer_size = buffer_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()

    return iterator

#convert_to_tfrecords('/n/pana/scratch/ravi/youtube-vos', '/n/pana/scratch/ravi/youtube-vos')
#65290
