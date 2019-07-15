#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/7/7
@Description:
"""
import numpy as np
from cv2 import imread, resize


def process_img(image):
    image = image.astype(np.float32)
    image = np.subtract(image, 128.)
    image = np.multiply(image, 1. / 128.)
    return image

#
# def _imread(image_name):
#     return imread(image_name)
#
#
# def _imresize(image_array, size):
#     return resize(image_array, size)
#
#
# def to_categorical(integer_classes, num_classes=2):
#     integer_classes = np.asarray(integer_classes, dtype='int')
#     num_samples = integer_classes.shape[0]
#     categorical = np.zeros((num_samples, num_classes))
#     categorical[np.arange(num_samples), integer_classes] = 1
#     return categorical
