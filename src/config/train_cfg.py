#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/7/6
@Description:
"""
import os
import time
import warnings
import tensorflow as tf
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

USE_EMOTION_DATASET = "RAF"     # RAF, FER2013
USE_EMOTION_MODEL = "VGG16"     # VGG16, MobileNet
IS_CONVERT2TFLITE = True

BATCH_SIZE = 32
EPOCHS = 200
LEARNING_RATE = 1e-4
INPUT_SHAPE = (64, 64, 3)   # MobileNet 96x96, VGG 64x64, FER2013 48x48
PATIENCE = 100
EMOTION_NUM_CLS = 7


_today = time.strftime("%Y%m%d", time.localtime())
_prefix = "../trained_models/emotion_models/" + USE_EMOTION_MODEL + USE_EMOTION_DATASET + '_' + _today
EMOTION_MODEL_NAME = _prefix + '.h5'
EMOTION_LOG_NAME = _prefix + '.log'
TFLITE_NAME = _prefix + '.tflite'


if USE_EMOTION_DATASET == "RAF":
    EMOTION_LABELS = ['Surprise', 'Fear', 'Disgust', 'Happy', 'Sad', 'Angry', 'Neutral']
elif USE_EMOTION_DATASET == "FER2013":
    EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
else:
    raise Exception("Only support emotion datasets RAF or FER2013 now!")
