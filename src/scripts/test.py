#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/7/13
@Description:
"""
import tensorflow as tf
from tensorflow.python.keras.utils.generic_utils import CustomObjectScope
from tensorflow.python.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.python.keras.models import load_model

# from tensorflow.keras.models import load_model
from tensorflow.contrib import lite
from ..config.train_cfg import *

converter = lite.TFLiteConverter.from_keras_model_file(EMOTION_MODEL_NAME)
tflite_model = converter.convert()
with open(TFLITE_NAME, "wb") as f:
    f.write(tflite_model)

# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from utils.data_manager import DataManager
# from utils.data_manager import split_raf_data
# import cv2
# import numpy as np
# data_generator = ImageDataGenerator(
#     rotation_range=30, horizontal_flip=True,
#     width_shift_range=0.1, height_shift_range=0.1,
#     zoom_range=0.2, shear_range=0.1,
#     channel_shift_range=0.5)
# # data_generator = ImageDataGenerator(width_shift_range=0.2)
# # loading dataset
# data_loader = DataManager("RAF", image_size=(64, 64))
# faces, emotions, usages = data_loader.get_data()
# # faces = process_img(faces)
# faces = np.multiply(faces, 1 / 255.)
# train_data, val_data = split_raf_data(faces, emotions, usages)
# #
# # print(train_data[0][0])
# # print(len(train_data))
# train_faces, train_emotions = train_data
# print(train_faces[0])
# data = data_generator.flow(train_faces, train_emotions, 3, shuffle=False)
# x = data.next()
# print("--------------------------------------")
# print(x[0][0])
#
# for i in range(0, 3):
#     cv2.imshow("origin"+str(i), cv2.cvtColor(train_faces[i], cv2.COLOR_RGB2BGR))
#     cv2.imshow("new"+str(i), cv2.cvtColor(x[0][i], cv2.COLOR_RGB2BGR))
# c = cv2.waitKey(0)
# if c & 0xFF == ord('q') or c & 0xFF == ord('Q'):
#     cv2.destroyAllWindows()
