#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/7/6
@Description:
"""
# import os
# from tensorflow.keras.models import load_model
# import tensorflow as tf
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.contrib import lite
from nets.choose_net import choose_net
from utils.data_manager import DataManager
from utils.data_manager import split_raf_data
from utils.preprocessor import process_img
from utils.plot import plot_log, plot_emotion_matrix
from config.train_cfg import *
from .evaluate import evaluate

# data generator
data_generator = ImageDataGenerator(
    rotation_range=30, horizontal_flip=True,
    width_shift_range=0.1, height_shift_range=0.1,
    zoom_range=0.2, shear_range=0.1,
    channel_shift_range=0.5)
#  channel_shift_range=50,
emotion_model = choose_net(USE_EMOTION_MODEL, INPUT_SHAPE, EMOTION_NUM_CLS)
sgd = optimizers.SGD(lr=LEARNING_RATE, decay=LEARNING_RATE/BATCH_SIZE, momentum=0.9, nesterov=True)
emotion_model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# callbacks
csv_logger = CSVLogger(EMOTION_LOG_NAME, append=False)
early_stop = EarlyStopping('val_loss', patience=PATIENCE)
reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1, patience=int(PATIENCE/4), verbose=1)
# model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(EMOTION_MODEL_NAME, 'val_loss', verbose=1,
                                   save_weights_only=False, save_best_only=True)
callbacks = [model_checkpoint, csv_logger, reduce_lr, early_stop]

# loading dataset
data_loader = DataManager(USE_EMOTION_DATASET, image_size=INPUT_SHAPE[:2])
faces, emotions, usages = data_loader.get_data()
faces = process_img(faces)
num_samples, num_classes = emotions.shape
train_data, val_data = split_raf_data(faces, emotions, usages)
train_faces, train_emotions = train_data

# if os.path.exists(EMOTION_MODEL_NAME):
#     emotion_net = load_model(EMOTION_MODEL_NAME)

emotion_model.fit_generator(data_generator.flow(train_faces, train_emotions, BATCH_SIZE),
                            steps_per_epoch=len(train_faces) / BATCH_SIZE,epochs=EPOCHS,
                            verbose=1, callbacks=callbacks, validation_data=val_data)


if IS_CONVERT2TFLITE:
    converter = lite.TFLiteConverter.from_keras_model_file(EMOTION_MODEL_NAME)
    tflite_model = converter.convert()
    with open(TFLITE_NAME, "wb") as f:
        f.write(tflite_model)

truth, prediction, accuracy = \
        evaluate(USE_EMOTION_DATASET, INPUT_SHAPE, EMOTION_MODEL_NAME)
plot_log(EMOTION_LOG_NAME)
plot_emotion_matrix(USE_EMOTION_DATASET, USE_EMOTION_MODEL, truth, prediction, accuracy)
