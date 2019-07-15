#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/7/14
@Description:
"""
import os
import time
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.contrib import lite
from ..utils.preprocessor import process_img
from ..utils.inference import load_image


# Load TFLite model and allocate tensors.
interpreter = lite.Interpreter(model_path="../trained_models/emotion_models/RAF_MobileNet_20190714.tflite")
image_path = "../images/happy1.jpg"     # 选择测试图片

rgb_image = load_image(image_path, color_mode='rgb')
face_image = cv2.resize(rgb_image, (64, 64))
face_image = process_img(face_image)
face_image = np.expand_dims(face_image, 0)

interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print(input_details)
print(output_details)


all_time = 0
for i in range(0,105):
    start = time.time()
    index = input_details[0]['index']
    interpreter.set_tensor(index, face_image)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    one_time = time.time()-start
    if i >= 5:
        all_time += one_time
print('output_data shape:', output_data.shape)
prediction = np.argmax(output_data)
print(prediction)
print("Average Time = ", all_time/100.)

