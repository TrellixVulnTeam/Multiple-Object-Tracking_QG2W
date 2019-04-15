import tensorflow as tf
import os, sys, io, glob
import time
import numpy as np
from PIL import Image
sys.path.append(os.path.realpath('./utils'))
from stream import VideoInputStream
from mask_rcnn_utils import mask_rcnn_get_full_masks, mask_rcnn_single_mask
import cv2
import random

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def validate_text(text):
    """If text is not str or unicode, then try to convert it to str."""
    if isinstance(text, str):
        return text
    elif isinstance(text, unicode):
        return text.encode('utf8', 'ignore')
    else:
        return str(text)

def create_tfrecord(frame_number, frame, bboxes, classes, scores, masks):

    height = frame.shape[0]
    width = frame.shape[1]

    filename = str(frame_number)

    scores = scores
    classes = classes.astype(np.int32)

    assert(masks.shape[0] == bboxes.shape[0] == scores.shape[0])

    mask_height = masks.shape[1]
    mask_width = masks.shape[2]
    num_objects = len(scores)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
        'image/filename': bytes_feature(filename),
        'image/source_id': bytes_feature(filename),
        'image/image': bytes_feature(frame.tobytes()),
        'image/mask_height': int64_feature(mask_height),
        'image/mask_width': int64_feature(mask_width),
        'image/num_objects': int64_feature(num_objects),
        'image/scores' : bytes_feature(scores.tobytes()),
        'image/masks' : bytes_feature(masks.tobytes()),
        'image/boxes' : bytes_feature(bboxes.tobytes()),
        'image/classes' : bytes_feature(classes.tobytes()),
    }))

    return tf_example

def segment_to_tfrecords(segment_path, detections_path, output_path,
                         height, width, stride, max_frames=None):

    writer = tf.python_io.TFRecordWriter(output_path)

    detections = {}
    if detections_path:
        detections = np.load(detections_path)[()]

    s = VideoInputStream(segment_path)
    frame_list = {}
    frame_idxs = []
    curr_pos = 0
    for frame in s:
        if (curr_pos in detections) and (curr_pos%stride==0):
            if (s.height > height) or (s.width > width):
                frame = cv2.resize(frame, (width, height))
            frame_list[curr_pos] = frame
            frame_idxs.append(curr_pos)
        curr_pos = curr_pos + 1
        if max_frames is not None and curr_pos > max_frames:
            break

    for idx in frame_idxs:
        frame = frame_list[idx]

        boxes, classes, scores, masks = detections[idx]

        if(boxes.size > 0):
            record = create_tfrecord(idx, frame, np.array(boxes),
                                     np.array(classes),
                                     np.array(scores),
                                     np.array(masks))
            writer.write(record.SerializeToString())
            print('Writing frame %d' %(idx))

    writer.close()

def get_dataset(filenames,
                buffer_size = 100,
                batch_size = 4,
                num_epochs = 50,
                num_threads = 4,
                height = 720,
                width = 1280,
                is_training = True):

    def parser(record):
        keys_to_features = {
                'image/height': tf.FixedLenFeature((), tf.int64),
                'image/width': tf.FixedLenFeature((), tf.int64),
                'image/filename': tf.FixedLenFeature((), tf.string),
                'image/source_id': tf.FixedLenFeature((), tf.string),
                'image/image': tf.FixedLenFeature((), tf.string),
                'image/mask_height': tf.FixedLenFeature((), tf.int64),
                'image/mask_width': tf.FixedLenFeature((), tf.int64),
                'image/num_objects': tf.FixedLenFeature((), tf.int64),
                'image/scores' : tf.FixedLenFeature((), tf.string),
                'image/masks' : tf.FixedLenFeature((), tf.string),
                'image/boxes' : tf.FixedLenFeature((), tf.string),
                'image/classes' : tf.FixedLenFeature((), tf.string),
        }

        parsed = tf.parse_single_example(record, keys_to_features)
        filename = parsed['image/filename']

        orig_height = tf.cast(parsed['image/height'], tf.int32)
        orig_width = tf.cast(parsed['image/width'], tf.int32)

        image = tf.decode_raw(parsed['image/image'], tf.uint8)

        image = tf.reshape(image, tf.stack([1, orig_height, orig_width, 3]))
        image = tf.image.resize_images(image, [height, width])
        image = tf.reshape(image, tf.stack([height, width, 3]))

        num_objects = tf.cast(parsed['image/num_objects'], tf.int32)
        mask_height = tf.cast(parsed['image/mask_height'], tf.int32)
        mask_width = tf.cast(parsed['image/mask_width'], tf.int32)

        scores = tf.decode_raw(parsed['image/scores'], tf.float32)
        scores = tf.reshape(scores, tf.stack([num_objects]))

        masks = tf.decode_raw(parsed['image/masks'], tf.float32)
        masks = tf.reshape(masks, tf.stack([num_objects, mask_height, mask_width]))

        boxes = tf.decode_raw(parsed['image/boxes'], tf.float64)
        boxes = tf.reshape(boxes, tf.stack([num_objects, 4]))

        classes = tf.decode_raw(parsed['image/classes'], tf.int32)
        classes = tf.reshape(classes, tf.stack([num_objects]))

        return image, boxes, scores, masks, classes, num_objects, filename

    if is_training:
        def interleave_fn(x):
            return tf.data.TFRecordDataset(x).map(parser, num_parallel_calls=num_threads).repeat(num_epochs)
        dataset = tf.data.Dataset.from_tensor_slices(filenames).\
                interleave(interleave_fn, cycle_length = len(filenames), block_length = 1)
        dataset = dataset.prefetch(buffer_size = buffer_size)
        dataset = dataset.shuffle(buffer_size = buffer_size)
    else:
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(parser, num_parallel_calls=num_threads)

    dataset = dataset.padded_batch(batch_size,
                padded_shapes=([None, None, None],
                               [None, None],
                               [None],
                               [None, None, None],
                               [None],
                               [],
                               []))

    iterator = dataset.make_one_shot_iterator()

    return iterator

def visualize_masks(labels, batch_size, image_shape,
                    num_classes = 5):

    masks = []
    for label in range(1, num_classes + 1):
        masks.append(labels == label)

    labels_vis = np.zeros((batch_size,
                           image_shape[0],
                           image_shape[1],
                           image_shape[2]), np.uint8)

    cmap = [[166, 206, 227],
            [178, 223, 138],
            [31,  120, 180],
            [51,  160,  44],
            [251, 154, 153],
            [227,  26,  28],
            [253, 191, 111],
            [255, 127,   0],
            [202, 178, 214],
            [106,  61, 154],
            [255, 255, 153],
            [177, 89,   40],
            [125, 125, 125]] # added a gray one. might not be perfect
    for i in range(num_classes):
        labels_vis[masks[i]] = cmap[i]

    return labels_vis

def batch_segmentation_masks(batch_size,
                             image_shape,
                             batch_boxes,
                             batch_classes,
                             batch_masks,
                             batch_scores,
                             batch_num_objects,
                             compute_weight_masks,
                             class_groups,
                             mask_threshold=0.5,
                             box_threshold=0.5,
                             scale_boxes=True):
    h = image_shape[0]
    w = image_shape[1]

    seg_masks = np.zeros((batch_size, h, w), np.uint8)
    weight_masks = np.zeros((batch_size, h, w), np.bool)

    class_remap = {}
    for g in range(len(class_groups)):
        for c in class_groups[g]:
            class_remap[c] = g + 1

    batch_boxes = batch_boxes.copy()

    if scale_boxes and len(batch_boxes.shape) == 3:
        batch_boxes[:, :, 0] = batch_boxes[:, :, 0] * h
        batch_boxes[:, :, 2] = batch_boxes[:, :, 2] * h
        batch_boxes[:, :, 1] = batch_boxes[:, :, 1] * w
        batch_boxes[:, :, 3] = batch_boxes[:, :, 3] * w

    batch_boxes = batch_boxes.astype(np.int32)

    for b in range(batch_size):
        N = batch_num_objects[b]
        if N == 0:
            continue
        boxes = batch_boxes[b, :N, :]
        masks = batch_masks[b, :N, :, :]
        scores = batch_scores[b, :N]
        classes = batch_classes[b, :N]

        for i in range(classes.shape[0]):
            if classes[i] in class_remap:
                classes[i] = class_remap[classes[i]]
            else:
                classes[i] = 0

        idx = classes > 0
        boxes = boxes[idx]
        masks = masks[idx]
        classes = classes[idx]
        scores = scores[idx]

        full_masks, box_masks = mask_rcnn_single_mask(boxes, classes,
                                                      scores, masks,
                                                      image_shape,
                                                      box_mask=compute_weight_masks,
                                                      box_threshold=box_threshold,
                                                      mask_threshold=mask_threshold)
        seg_masks[b] = full_masks
        weight_masks[b] = box_masks

    return seg_masks, weight_masks

def test_dataset():
    input_path = '/data/ravi/static_cam_dataset/samui_murphys'
    record_files = glob.glob(os.path.join(input_path, '*.tfrecords'))

    iterator = get_dataset(record_files, batch_size=4, buffer_size=50, num_threads=8)

    tensors = iterator.get_next()

    people_cls = [1]
    seg_tensor_people = tf.py_func(batch_segmentation_masks, [4, (720, 1280, 3),
                         tensors[2], tensors[5], tensors[4],
                         tensors[6], people_cls], tf.bool)

    vehicle_cls = [2, 3, 4, 6, 7, 8, 9]
    seg_tensor_vehicle = tf.py_func(batch_segmentation_masks, [4, (720, 1280, 3),
                         tensors[2], tensors[5], tensors[4],
                         tensors[6], vehicle_cls], tf.bool)

    seg_tensor_vehicle.set_shape([4, 720, 1280])
    seg_tensor_people.set_shape([4, 720, 1280])

    people_labels = tf.cast(seg_tensor_people, tf.int32) * 1
    vehicle_labels = tf.cast(seg_tensor_vehicle, tf.int32) * 2

    labels = tf.zeros(people_labels.shape, tf.int32)
    labels = tf.where(tf.cast(seg_tensor_people, tf.int32) > 0, people_labels, labels)
    labels = tf.where(tf.cast(seg_tensor_vehicle, tf.int32) > 0, vehicle_labels, labels)

    with tf.Session() as sess:
        for i in range(100):
            tic = time.time()
            vals, people_mask, vehicle_mask, label_vals = sess.run([tensors, seg_tensor_people,
                                                                    seg_tensor_vehicle, labels])
            print(people_mask.shape, vehicle_mask.shape, label_vals.shape)
            toc = time.time()
            print('Processing', i, toc - tic)
