from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time
import glob
import random

import numpy as np
import tensorflow as tf
from PIL import Image
from StringIO import StringIO

import input_preprocess

def num_examples_per_epoch(split='train'):
    if split == 'train':
        return 2975
    if split == 'val':
        return 500
    if split == 'extra':
        return 19997

def num_classes(meta = False):
    if meta:
        return 7
    else:
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
               7:0, 8:1, 9:255, 10:255, 11:2, 12:3, 13:4, 14:255,
               15:255, 16:255, 17:5, 18:255, 19:6, 20:7, 21:8, 22:9,
               23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255, 30:255,
               31:16, 32:17, 33:18, 255:255 }

trainid2group = { 0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 3, 9: 3, 10: 4, 11: 5, 12: 5,
                  13: 6, 14: 6, 15: 6, 16: 6, 17: 6, 18: 6 }

groupid2name = {0: 'flat', 1: 'construction', 2: 'object', 3: 'nature', 4: 'sky',
                5: 'human', 6: 'vehicle'}

id2name = {0: 'road', 1: 'sidewalk', 2:'building', 3:'wall', 4:'fence', 5:'pole', 6:'traffic_light',
           7:'traffic_sign', 8: 'vegetation', 9: 'terrain', 10: 'sky', 11: 'person', 12: 'rider', 13: 'car',
           14:'truck', 15: 'bus', 16:'train', 17:'motorcycle', 18:'bicycle'}

id2color = [(128, 64, 128), (244, 35, 232), ( 70, 70, 70), (102,102,156), (190,153,153),
            (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
            ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
            (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32)]

class_weights = [2.8149201869965,
                 6.9850029945374,
                 3.7890393733978,
                 9.9428062438965,
                 9.7702074050903,
                 9.5110931396484,
                 10.311357498169,
                 10.026463508606,
                 4.6323022842407,
                 9.5608062744141,
                 7.8698215484619,
                 9.5168733596802,
                 10.373730659485,
                 6.6616044044495,
                 10.260489463806,
                 10.287888526917,
                 10.289801597595,
                 10.405355453491,
                 10.138095855713]

def convert_self_to_tfrecords(data_path, out_path, encoded = True):
    data = glob.glob(data_path + '/*image*')
    data = [ item.split('/')[-1].split('_')[0] for item in data]

    tfrecord_file_name = os.path.join(out_path, 'cityscapes_self_extra.tfrecords')
    writer = tf.python_io.TFRecordWriter(tfrecord_file_name)

    city = 'all'
    for frame in data:
        print(frame)
        im_file_name = '{}/{}_image.png'.format(data_path, frame)
        im_data = tf.gfile.FastGFile(im_file_name, 'r').read()

        label_file_name = '{}/{}_train_id.png'.format(data_path, frame)
        label_data = tf.gfile.FastGFile(label_file_name, 'r').read()

        im = np.asarray(Image.open(StringIO(im_data)))
        rows, cols, depth = im.shape[0], im.shape[1], im.shape[2]

        if not encoded:
            im_data = np.asarray(Image.open(StringIO(im_data))).tostring()
            label_data = np.asarray(Image.open(StringIO(label_data)))
            label_data = label_data.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
                                    'height': _int64_feature(rows),
                                    'width' : _int64_feature(cols),
                                    'depth' : _int64_feature(depth),
                                    'image' : _bytes_feature(im_data),
                                    'labels': _bytes_feature(label_data),
                                    'city'  : _bytes_feature(city),
                                    'frame' : _bytes_feature(frame),
                                    'format': _bytes_feature('png'),
                                   }))

        writer.write(example.SerializeToString())
    writer.close()

def convert_xception_to_tfrecords(data_path, pred_path, out_path, encoded = True):
    data = glob.glob(data_path + '/gtCoarse/train_extra/*/*_labelIds*')

    random.seed(0)
    random.shuffle(data)

    data = [(item.split('/')[-2],
             '_'.join(item.split('/')[-1].split('.')[0].split('_')[:-2])) for item in data]

    tfrecord_file_name = os.path.join(out_path, 'cityscapes_xception_extra.tfrecords')
    writer = tf.python_io.TFRecordWriter(tfrecord_file_name)

    for city, frame in data:
        print(frame)
        im_file_name = '{}/leftImg8bit/train_extra/{}/{}_leftImg8bit.png'.format(data_path, city, frame)
        im_data = tf.gfile.FastGFile(im_file_name, 'r').read()

        label_file_name = '{}/raw_segmentation_results/{}.png'.format(pred_path, frame)
        label_data = tf.gfile.FastGFile(label_file_name, 'r').read()

        im = np.asarray(Image.open(StringIO(im_data)))
        rows, cols, depth = im.shape[0], im.shape[1], im.shape[2]

        if not encoded:
            im_data = np.asarray(Image.open(StringIO(im_data))).tostring()
            label_data = np.asarray(Image.open(StringIO(label_data)))
            label_data.setflags(write=1)
            for k, v in id2trainid.iteritems():
                label_data[k == label_data] = v
            label_data = label_data.tostring()
        else:
            label_data = np.asarray(Image.open(StringIO(label_data)))
            label_data.setflags(write=1)
            for k, v in id2trainid.iteritems():
                label_data[k == label_data] = v
            label_out = StringIO()
            Image.fromarray(label_data).save(label_out, 'png')
            label_data = label_out.getvalue()

        example = tf.train.Example(features=tf.train.Features(feature={
                                    'height': _int64_feature(rows),
                                    'width' : _int64_feature(cols),
                                    'depth' : _int64_feature(depth),
                                    'image' : _bytes_feature(im_data),
                                    'labels': _bytes_feature(label_data),
                                    'city'  : _bytes_feature(city),
                                    'frame' : _bytes_feature(frame),
                                    'format': _bytes_feature('png'),
                                   }))

        writer.write(example.SerializeToString())
    writer.close()

def convert_coarse_to_tfrecords(data_path, out_path, encoded = True):
    data = glob.glob(data_path + '/gtCoarse/train_extra/*/*_labelIds*')
    data = [(item.split('/')[-2],
             '_'.join(item.split('/')[-1].split('.')[0].split('_')[:-2])) for item in data]

    tfrecord_file_name = os.path.join(out_path, 'cityscapes_extra.tfrecords')
    writer = tf.python_io.TFRecordWriter(tfrecord_file_name)

    for city, frame in data:
        print(frame)
        im_file_name = '{}/leftImg8bit/train_extra/{}/{}_leftImg8bit.png'.format(data_path, city, frame)
        im_data = tf.gfile.FastGFile(im_file_name, 'r').read()

        label_file_name = '{}/gtCoarse/train_extra/{}/{}_gtCoarse_labelIds.png'.format(data_path, city, frame)
        label_data = tf.gfile.FastGFile(label_file_name, 'r').read()

        im = np.asarray(Image.open(StringIO(im_data)))
        rows, cols, depth = im.shape[0], im.shape[1], im.shape[2]

        if not encoded:
            im_data = np.asarray(Image.open(StringIO(im_data))).tostring()
            label_data = np.asarray(Image.open(StringIO(label_data)))
            label_data.setflags(write=1)
            for k, v in id2trainid.iteritems():
                label_data[k == label_data] = v
            label_data = label_data.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
                                    'height': _int64_feature(rows),
                                    'width' : _int64_feature(cols),
                                    'depth' : _int64_feature(depth),
                                    'image' : _bytes_feature(im_data),
                                    'labels': _bytes_feature(label_data),
                                    'city'  : _bytes_feature(city),
                                    'frame' : _bytes_feature(frame),
                                    'format': _bytes_feature('png'),
                                   }))

        writer.write(example.SerializeToString())
    writer.close()

def convert_to_tfrecords(split, data_path, out_path, encoded = True):
    assert(split in ['train', 'test', 'val'])
    data = open('{}/SegFine/{}.txt'.format(data_path, split)).read().splitlines()
    data = [(item.split('/')[0], item.split('/')[1]) for item in data]

    tfrecord_file_name = os.path.join(out_path, 'cityscapes_{}.tfrecords'.format(split))
    writer = tf.python_io.TFRecordWriter(tfrecord_file_name)

    for city, frame in data:
        im_file_name = '{}/leftImg8bit/{}/{}/{}_leftImg8bit.png'.format(data_path, split, city, frame)
        im_data = tf.gfile.FastGFile(im_file_name, 'r').read()

        label_file_name = '{}/gtFine/{}/{}/{}_gtFine_labelIds.png'.format(data_path, split, city, frame)
        label_data = tf.gfile.FastGFile(label_file_name, 'r').read()

        im = np.asarray(Image.open(StringIO(im_data)))
        rows, cols, depth = im.shape[0], im.shape[1], im.shape[2]

        if not encoded:
            im_data = np.asarray(Image.open(StringIO(im_data))).tostring()
            label_data = np.asarray(Image.open(StringIO(label_data)))
            label_data.setflags(write=1)
            for k, v in id2trainid.iteritems():
                label_data[k == label_data] = v
            label_data = label_data.tostring()

        example = tf.train.Example(features=tf.train.Features(feature={
                                    'height': _int64_feature(rows),
                                    'width' : _int64_feature(cols),
                                    'depth' : _int64_feature(depth),
                                    'image' : _bytes_feature(im_data),
                                    'labels': _bytes_feature(label_data),
                                    'city'  : _bytes_feature(city),
                                    'frame' : _bytes_feature(frame),
                                    'format': _bytes_feature('png'),
                                   }))

        writer.write(example.SerializeToString())
    writer.close()

def get_dataset(filename,
                buffer_size = 100,
                batch_size = 4,
                num_epochs = 50,
                encoded = True,
                num_threads = 8,
                data_augment = False):

    dataset = tf.data.TFRecordDataset([filename])
    def parser(record):
        keys_to_features = {'height': tf.FixedLenFeature((), tf.int64),
                            'width' : tf.FixedLenFeature((), tf.int64),
                            'depth' : tf.FixedLenFeature((), tf.int64),
                            'image' : tf.FixedLenFeature((), tf.string),
                            'labels' : tf.FixedLenFeature((), tf.string),
                            'city' : tf.FixedLenFeature((), tf.string),
                            'frame' : tf.FixedLenFeature((), tf.string),
                            'format' : tf.FixedLenFeature((), tf.string),
                           }
        parsed = tf.parse_single_example(record, keys_to_features)

        if encoded:
            image = tf.image.decode_png(parsed['image'])
            labels = tf.image.decode_png(parsed['labels'])
        else:
            image = tf.decode_raw(parsed['image'], tf.uint8)
            labels = tf.decode_raw(parsed['labels'], tf.uint8)

        #height = tf.cast(parsed['height'], tf.int32)
        #width = tf.cast(parsed['width'], tf.int32)
        #depth = tf.cast(parsed['depth'], tf.int32)

        image_shape = tf.stack([1024, 2048, 3])
        labels_shape = tf.stack([1024, 2048, 1])

        image = tf.reshape(image, image_shape)
        labels = tf.reshape(labels, labels_shape)

        if data_augment:
            _, image, labels = \
                input_preprocess.preprocess_image_and_label(image, labels,
                                                            1024, 2048,
                                                            min_scale_factor=0.5,
                                                            max_scale_factor=2,
                                                            scale_factor_step_size=0.25)

        return image, labels

    dataset = dataset.map(parser, num_parallel_calls=num_threads)
    dataset = dataset.shuffle(buffer_size = buffer_size)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)

    iterator = dataset.make_one_shot_iterator()
    #iterator = dataset.make_initializable_iterator()

    return iterator

if __name__ == '__main__':

    iterator = \
        get_dataset('/data/ravi/cityscapes/cityscapes_train.tfrecords', encoded = False)
    batch_image, batch_labels = iterator.get_next()

    with tf.Session() as sess:
        for i in range(500):
            start = time.time()
            image_vals, labels_vals = sess.run([batch_image, batch_labels])
            end = time.time()
            print(i, image_vals.shape, labels_vals.shape, end - start)
