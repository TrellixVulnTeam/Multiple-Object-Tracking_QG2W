import sys
import os
import cv2
import glob
import numpy as np
from PIL import Image

class ImageSequence:
    def __init__(self, img_dir, img_regex, loop = False):
        img_list = glob.glob(os.path.join(img_dir, img_regex))
        self.img_list = sorted(img_list)
        self.pos = 0
        self.sequence_len = len(img_list)
        self.loop = loop

    def __next__(self):
        if self.pos >= self.sequence_len:
            if self.loop:
                self.pos = 0
            else:
                raise StopIteration()

        im = cv2.imread(self.img_list[self.pos])
        name = self.img_list[self.pos]
        self.pos = self.pos + 1
        return im, name

    def __iter__(self):
        return self

    next = __next__
