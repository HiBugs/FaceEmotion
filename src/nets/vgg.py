#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/5/30
@Description:
"""
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization
from tensorflow.keras.layers import Activation, Convolution2D, Conv2D
from tensorflow.keras.layers import MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D


def vgg16(input_shape, num_classes, weights_path=None, pooling='avg'):
    # 构造VGG16模型
    model = Sequential()

    # Block 1
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'))
    # model.add(BatchNormalization(name='bn_1'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'))

    # Block 2
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'))
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'))
    # model.add(BatchNormalization(name='bn_2'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'))

    # Block 3
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'))
    model.add(Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'))
    # model.add(BatchNormalization(name='bn_3'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'))

    # Block 4
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'))
    # model.add(BatchNormalization(name='bn_4'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'))

    # Block 5
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'))
    model.add(Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'))
    # model.add(BatchNormalization(name='bn_5'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool'))

    if weights_path:
        model.load_weights(weights_path)

    out = model.get_layer('block5_pool').output

    if pooling is None:
        out = Flatten(name='flatten')(out)
        out = Dense(512, activation='relu', kernel_initializer='he_normal', name='fc')(out)
        out = Dropout(0.5)(out)
        # out = Dense(512, activation='relu', kernel_initializer='he_normal', name='fc2')(out)
        # out = Dropout(0.5)(out)
    elif pooling == 'avg':
        out = GlobalAveragePooling2D(name='global_avg_pool')(out)
    elif pooling == 'max':
        out = GlobalMaxPooling2D(name='global_max_pool')(out)

    out = Dense(num_classes, activation='softmax', kernel_initializer='he_normal', name='predict')(out)

    model = Model(model.input, out)

    return model
