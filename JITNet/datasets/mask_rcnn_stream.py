import os
import cv2
import numpy as np
import time
import tensorflow as tf
from random import randint
from itertools import chain

class MaskRCNNStream:
    def __init__(self, video_stream_path, detections_path,
                 start_frame=0, num_frames=None, stride=1,
                 loop=True):
        assert(os.path.isfile(video_stream_path))
        assert(os.path.isfile(detections_path))
        self.cap = cv2.VideoCapture(video_stream_path)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.rate = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.length = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.num_frames = num_frames

        self.detections_path = detections_path
        self.detections = None
        self.stride = stride
        self.loop = loop

        assert(start_frame >= 0)
        self.start_frame = start_frame
        self.end_frame = self.length

        # Seek to the start frame
        if self.start_frame > 0:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    def __next__(self):
        if self.detections is None:
            self.detections = np.load(self.detections_path)[()]
            self.labeled_frames = list(self.detections.keys())
            self.num_labeled_frames = len(self.labeled_frames)
            if self.num_frames is not None:
                assert(self.start_frame + self.num_frames <= self.length)
                self.end_frame = (start_frame + self.num_frames) - 1

        frame = None
        boxes = None
        classes = None
        scores = None
        masks = None
        labels_not_found = True
        while labels_not_found:
            frame_id = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
            ret, frame = self.cap.read()

            if (not ret) or (frame_id >= self.end_frame - 1):
                if self.loop:
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                    frame_id = self.start_frame
                else:
                    self.detections = None
                    raise StopIteration

            if frame_id in self.detections and frame_id%self.stride==0:
                boxes, classes, scores, masks = self.detections[frame_id]
                labels_not_found = False

        return frame, boxes, classes, scores, masks, scores.shape[0], frame_id

    def __iter__(self):
        return self

    def reset(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)

    next = __next__


class MaskRCNNMultiStream:
    def __init__(self, video_paths, detections_paths,
                 start_frame=0, stride=1):
        self.streams = []
        self.stream_idx = 0
        self.num_streams = len(video_paths)
        print(video_paths)
        for d in range(len(video_paths)):
            input_stream = MaskRCNNStream(video_paths[d], detections_paths[d],
                                          start_frame=start_frame, stride=stride)
            self.streams.append(input_stream)

    def __next__(self):
        self.stream_idx = (self.stream_idx + 1) % self.num_streams
        frame, boxes, classes, scores, masks, scores.shape[0], frame_id

        return self.streams[self.stream_idx].__next__()

    def __iter__(self):
        return self

    next = __next__

class MaskRCNNSequenceStream:
    def __init__(self, video_paths, detections_paths,
                 start_frame=0, stride=1):
        self.streams = []
        self.stream_idx = 0
        self.num_streams = len(video_paths)
        self.rate = 0
        print(video_paths)
        for d in range(len(video_paths)):
            input_stream = MaskRCNNStream(video_paths[d], detections_paths[d],
                                          start_frame=start_frame, stride=stride,
                                          loop=False)
            self.streams.append(input_stream)
            #print(self.rate, input_stream.rate)
            if self.rate == 0:
                self.rate = input_stream.rate
            #else:
            #    assert(self.rate == input_stream.rate)
        self.seq_stream = chain(*(self.streams))

    def __next__(self):
        return next(self.seq_stream)

    def __iter__(self):
        return self

    next = __next__

def get_dataset(video_paths, detections_paths, batch_size, start_frame=0, stride=1):

    input_streams = MaskRCNNMultiStream(video_paths, detections_paths, start_frame=start_frame, stride=stride)
    dataset = tf.data.Dataset.from_generator(lambda: input_streams, (tf.uint8,
                                                                     tf.float32,
                                                                     tf.int32,
                                                                     tf.float32,
                                                                     tf.float32,
                                                                     tf.int32,
                                                                     tf.int32))
    dataset = dataset.padded_batch(batch_size,
                padded_shapes=([None, None, None],
                               [None, None],
                               [None],
                               [None],
                               [None, None, None],
                               [],
                               []))

    iterator = dataset.make_one_shot_iterator()

    return iterator

def test_create():
    mask_rcnn_stream = MaskRCNNStream('/n/scanner/ravi/static_cam_dataset/bryant_park/bryant_park_shop_1-0.mp4',
            '/n/scanner/ravi/static_cam_dataset/bryant_park/mask_rcnn_1_bryant_park_shop_1-0.npy', num_frames=100)
    count = 0
    start = time.time()
    for s in mask_rcnn_stream:
        frame, boxes, classes, scores, masks, num_objects = s
        count = count + 1
    end = time.time()
    print(count, (end - start)/count)
