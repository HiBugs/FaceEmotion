#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/7/6
@Description:
"""
from scipy.io import loadmat
from PIL import Image
import pandas as pd
import numpy as np
from random import shuffle
import os
import cv2


class DataManager(object):
    def __init__(self, dataset_name, image_size=(64, 64)):
        self.dataset_name = dataset_name
        self.image_size = image_size
        if self.dataset_name == 'FER2013':
            self.dataset_path = '../datasets/fer2013/fer2013.csv'
        elif self.dataset_name == 'RAF':
            self.dataset_path = '../datasets/RAFdb/'
        else:
            raise Exception('Please input RAF or FER2013!')

    def get_data(self):
        if self.dataset_name == 'FER2013':
            ground_truth_data = self._load_fer2013()
        elif self.dataset_name == 'RAF':
            ground_truth_data = self._load_raf()
        return ground_truth_data

    def _load_fer2013(self):
        data = pd.read_csv(self.dataset_path)
        pixels = data['pixels'].tolist()
        width, height = 48, 48
        faces = []
        for pixel_sequence in pixels:
            face = [int(pixel) for pixel in pixel_sequence.split(' ')]
            face = np.asarray(face).reshape(width, height)
            face = cv2.resize(face.astype('uint8'), self.image_size)
            faces.append(face.astype('float32'))
        faces = np.asarray(faces)
        faces = np.expand_dims(faces, -1)
        # emotions = pd.get_dummies(data['emotion']).as_matrix()
        emotions = pd.get_dummies(data['emotion']).values
        return faces, emotions

    def _load_raf(self):
        with open(self.dataset_path + 'label.txt', 'r+', encoding='utf8') as f:
            images_list = f.readlines()
        label_list, pixels_list, usage_list = [], [], []
        for num, image in enumerate(images_list):
            image_name, image_label = image.split(' ')
            label_list.append(str(image_label).replace('\n', ''))
            pixels_list.append(self._read_one_image(os.path.join(self.dataset_path, image_name)).astype('float32'))
            usage_list.append(image_name.split('_')[0])
        faces = np.asarray(pixels_list)
        emotions = pd.get_dummies(label_list).values
        return faces, emotions, usage_list

    def _read_one_image(self, name):
        image = cv2.imread(name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
        # image = Image.open(name).convert("RGB")
        # # image = image.convert('L')
        image = cv2.resize(image, self.image_size)
        image = np.array(image)
        # image = image.reshape([self.image_size[0], self.image_size[0], 3])
        # image = image.astype(np.float32)
        # image = np.subtract(image, 128.)
        # image = np.multiply(image, 1. / 128.)
        return image


def get_labels(dataset_name):
    if dataset_name == 'FER2013':
        return {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy',
                4: 'sad', 5: 'surprise', 6: 'neutral'}
    elif dataset_name == 'RAF':
        return {1: 'Surprise', 2: 'Fear', 3: 'Disgust', 4: 'Happiness',
                5: 'Sadness', 6: 'Anger', 7: 'Neutral'}
    else:
        raise Exception('Invalid dataset name')


def get_class_to_arg(dataset_name):
    if dataset_name == 'FER2013':
        return {'angry': 0, 'disgust': 1, 'fear': 2, 'happy': 3, 'sad': 4,
                'surprise': 5, 'neutral': 6}
    elif dataset_name == 'RAF':
        return {'Surprise': 1, 'Fear': 2, 'Disgust': 3, 'Happiness': 4, 'Sadness': 5,
                'Anger': 6, 'Neutral': 7}
    else:
        raise Exception('Invalid dataset name')


def split_raf_data(x, y, use):
    train_x, train_y, val_x, val_y = [], [], [], []

    for i, u in enumerate(use):
        if u == "train":
            train_x.append(x[i])
            train_y.append(y[i])
        else:
            val_x.append(x[i])
            val_y.append(y[i])

    train_x, train_y, val_x, val_y = np.asarray(train_x), np.asarray(train_y), np.asarray(val_x), np.asarray(val_y)

    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data


def split_data(x, y, validation_split=.2):
    num_samples = len(x)
    num_train_samples = int((1 - validation_split)*num_samples)
    train_x = x[:num_train_samples]
    train_y = y[:num_train_samples]
    val_x = x[num_train_samples:]
    val_y = y[num_train_samples:]
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)
    return train_data, val_data
