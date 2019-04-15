from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image
import os
import numpy as np
import sys
import cv2

# Mask R-CNN utilities

sys.path.append(os.path.realpath('./datasets'))
sys.path.append(os.path.realpath('./utils'))

from mask_rcnn_tfrecords import get_dataset, batch_segmentation_masks,\
                                visualize_masks
from mask_rcnn_stream import MaskRCNNSequenceStream

import video_distillation
from video_distillation import sequence_to_class_groups_stable


"""Dataset class for OSVOS, using the same interface as the original."""
class OSVOS_Dataset:
    def __init__(self, sequence, dataset_dir, sequence_limit, stride, height, width, class_index, start_frame):
        """Initialize the Dataset object
        Args:
            sequence: sequence name
            dataset_dir: Absolute path to dataset root
            sequence_limit: maximum number of video sequence chunks to load
            stride: stride to run MRCNN teacher
            height: height of full size output images
            width: width of full size output images
            class_index: one-indexed class to segment
        """
        self.height = height
        self.width = width
        self.class_index = class_index
        self.stride = stride
        self.sequence_limit = sequence_limit

        # Load the sequence using JITNet utilities
        print('Initializing dataset...')

        # assemble video files and detection paths
        video_files = []
        detections_paths = []

        sequence_to_video_list = \
                video_distillation.get_sequence_to_video_list(
                        dataset_dir, dataset_dir,
                        video_distillation.video_sequences_stable)
        assert(sequence in sequence_to_video_list)
        sequence_path = os.path.join(dataset_dir, sequence)

        assert(os.path.isdir(sequence_path))
        assert(sequence in video_distillation.sequence_to_class_groups_stable)

        num_sequences = 0
        for s in sequence_to_video_list[sequence]:
            video_files.append(os.path.join(sequence_path, s[0]))
            detections_paths.append(os.path.join(sequence_path, s[1]))
            num_sequences = num_sequences + 1
            if num_sequences >= sequence_limit:
                break

        self.class_groups = sequence_to_class_groups_stable[sequence]

        print(video_files)
        print(detections_paths)
        print(self.class_groups)

        self.class_groups = [ [video_distillation.detectron_classes.index(c) for c in g] \
                         for g in self.class_groups ]
        self.num_classes = len(self.class_groups) + 1

        self.input_streams = MaskRCNNSequenceStream(video_files,
                                                    detections_paths,
                                                    start_frame=start_frame,
                                                    stride=1)

        img, label = self._stream_next(self.input_streams)
        # these are also used for the first test batch
        self.first_img = img
        self.first_label = label

        self._augment_data(img, label)

        # Init parameters
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = 6 # 3 scales, with a flip for each scale
        self.test_size = stride
        self.train_idx = np.arange(self.train_size)
        np.random.shuffle(self.train_idx)

        print('Done initializing Dataset.')

    def _augment_data(self, img, label):
        self.images_train = []
        self.labels_train = []
        data_aug_scales = [0.5, 0.8, 1]

        for scale in data_aug_scales:
            img_size = (int(self.height * scale), int(self.width * scale))
            img_sc = cv2.resize(img, img_size)
            label_sc = cv2.resize(label, img_size)
            self.images_train.append(np.array(img_sc, dtype=np.uint8))
            self.labels_train.append(np.array(label_sc, dtype=np.uint8))
            # add flip
            img_sc_fl = np.fliplr(img_sc).astype(np.uint8)
            label_sc_fl = np.fliplr(label_sc).astype(np.uint8)
            self.images_train.append(img_sc_fl)
            self.labels_train.append(label_sc_fl)


    def _stream_next(self, stream):
        # grab a single image and mask from input_streams
        frame, boxes, classes, scores, masks, num_objects, frame_id = next(stream)

        img = cv2.resize(frame, (self.width, self.height))

        boxes = np.expand_dims(boxes, axis=0)
        classes = np.expand_dims(classes, axis=0)
        scores = np.expand_dims(scores, axis=0)
        masks = np.expand_dims(masks, axis=0)
        num_objects = np.expand_dims(num_objects, axis=0)

        labels_vals, _ = batch_segmentation_masks(1,
                                                  (self.height, self.width),
                                                  boxes, classes, masks, scores,
                                                  num_objects, True,
                                                  self.class_groups)

        labels_val = np.reshape(labels_vals, (self.height, self.width))
        # only consider one class label
        labels_mask = (labels_val == self.class_index)
        label = np.zeros((self.height, self.width), dtype=np.uint8)
        label[labels_mask] = 255

        return img, label


    def next_batch(self, batch_size, phase):
        """Get next batch of image (path) and labels
        Args:
            phase: 'train' or 'test', starts with one train
        Returns:
            images: List of Numpy arrays of the images
            labels: List of Numpy arrays of the labels
        """
        if batch_size != 1:
            raise ValueError('batch size only 1')

        if phase == 'train':
            # return from the premade list
            index = self.train_idx[self.train_ptr]
            self.train_ptr += 1
            self.train_ptr = self.train_ptr % self.train_size
            if self.train_ptr == 0:
                np.random.shuffle(self.train_idx)
            return [self.images_train[index]], [self.labels_train[index]]

        else:
            # read from the stream and return
            if self.test_ptr == 0:
                self.test_ptr += 1
                return [self.first_img], [self.first_label]
            else:
                img, label = self._stream_next(self.input_streams)
                if self.test_ptr >= self.test_size:
                    raise StopIteration('next_batch test should not be called more than stride times.')
                self.test_ptr += 1
                return [img], [label]


    def reset_cycle(self):
        """Get ready for the next cycle of data."""
        # grab a new test frame and perform augmentation.
        img, label = self._stream_next(self.input_streams)
        # these are also used for the first test batch
        self.first_img = img
        self.first_label = label

        self._augment_data(img, label)

        # reset the pointers and variables
        self.train_ptr = 0
        self.test_ptr = 0
        self.train_size = 6 # 3 scales, with a flip for each scale
        self.test_size = self.stride
        self.train_idx = np.arange(self.train_size)
        np.random.shuffle(self.train_idx)

    def get_train_size(self):
        return self.train_size

    def get_test_size(self):
        return self.test_size
