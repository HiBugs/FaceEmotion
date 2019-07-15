#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/7/6
@Description:
"""
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils.data_manager import DataManager
from utils.data_manager import split_raf_data
from utils.plot import plot_log, plot_emotion_matrix, plot_progress
from utils.preprocessor import process_img
from config.train_cfg import INPUT_SHAPE, USE_EMOTION_DATASET,USE_EMOTION_MODEL

# import warnings
# warnings.filterwarnings("ignore")
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# gpu_options = tf.GPUOptions(allow_growth=True)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


def evaluate(dataset, input_shape, model_path):
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        raise Exception("The model doesn't exist!")

    truth, prediction = [], []
    print("load data ...")
    if dataset == 'RAF':
        # loading dataset
        data_loader = DataManager(dataset, image_size=input_shape[:2])
        faces, emotions, usages = data_loader.get_data()
        faces = process_img(faces)
        train_data, val_data = split_raf_data(faces, emotions, usages)
        data, label = val_data
        count = len(label)
        correct = 0
        for i, d in enumerate(data):
            if i % 200 == 0:
                plot_progress(i, count)
            d = np.expand_dims(d, 0)
            emotion_values = model.predict(d)
            emotion_label_arg = np.argmax(emotion_values)
            p = int(emotion_label_arg)
            t = int(np.argmax(label[i]))
            if p == t:
                correct += 1
            prediction.append(p)
            truth.append(t)
        accuracy = correct/float(count)
        print(correct, count, accuracy)

    else:
        raise Exception("RAF only!")

    return truth, prediction, accuracy


if __name__ == '__main__':
    dataset = 'RAF'     # USE_EMOTION_DATASET
    model_name = "VGG16"        # USE_EMOTION_MODEL
    truth, prediction, accuracy = \
        evaluate(USE_EMOTION_DATASET, INPUT_SHAPE, '../trained_models/emotion_models/VGG16_Dense_RAF_20190714.h5')
    plot_emotion_matrix(USE_EMOTION_DATASET, USE_EMOTION_MODEL, truth, prediction, accuracy)
    plot_log('../trained_models/emotion_models/VGG16_Dense_RAF_20190714.log')
