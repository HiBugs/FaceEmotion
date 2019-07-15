#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/5/29
@Description:
"""
from tensorflow.keras.utils import plot_model
from tensorflow.keras.metrics import top_k_categorical_accuracy
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Activation, BatchNormalization
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from tensorflow.keras.layers import add, Input, Dense, Dropout, Flatten


RESNET_CFG = {
    'resnet_18': [2, 2, 2, 2],
    'resnet_34': [3, 4, 6, 3],
    'resnet_50': [3, 4, 6, 3],
    'resnet_101': [3, 4, 23, 3],
    'resnet_152': [3, 8, 36, 3]
}


def model_block(name):
    if name == 'resnet18' or name == 'resnet34':
        return [64], identity_block
    else:
        return [64, 64, 256], bottleneck_block


def Conv2d_bN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def identity_block(inpt, nb_filter, strides=(1, 1), with_conv_shortcut=False):
    f = nb_filter[0]
    x = Conv2d_bN(inpt, nb_filter=f, kernel_size=3, strides=strides, padding='same')
    x = Conv2d_bN(x, nb_filter=f, kernel_size=3, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_bN(inpt, nb_filter=f, strides=strides, kernel_size=3)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def bottleneck_block(inpt, nb_filters, strides=(1, 1), with_conv_shortcut=False):
    f1, f2, f3 = nb_filters
    x = Conv2d_bN(inpt, nb_filter=f1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_bN(x, nb_filter=f2, kernel_size=3, padding='same')
    x = Conv2d_bN(x, nb_filter=f3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_bN(inpt, nb_filter=f3, strides=strides, kernel_size=1)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def resnet(name, input_shape, classes):
    filters, block = model_block(name)
    inpt = Input(shape=input_shape)

    x = Conv2d_bN(inpt, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    for i, cnt in enumerate(RESNET_CFG[name]):  # cnt为每个子模块重复次数
        for k in range(cnt):
            isshortcut = True if k == 0 else False      # 如果是每个子模块的第一层，则
            strides = (2, 2) if (i!=0and k==0) else (1, 1)  # 如果是非第一个子模块的第一层, strides=(2, 2)
            # print(filters, strides, isshortcut)
            x = block(x, filters, strides, isshortcut)
        filters = [i * 2 for i in filters]

    x = AveragePooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model

def resnet_34(width,height,channel,classes):
    inpt = Input(shape=(width, height, channel))
    x = ZeroPadding2D((3, 3))(inpt)

    #conv1
    x = Conv2d_bN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    #conv2_x
    x = identity_block(x, nb_filter=[64])
    x = identity_block(x, nb_filter=[64])
    x = identity_block(x, nb_filter=[64])

    #conv3_x
    x = identity_block(x, nb_filter=[128], strides=(2, 2), with_conv_shortcut=True)
    x = identity_block(x, nb_filter=[128])
    x = identity_block(x, nb_filter=[128])
    x = identity_block(x, nb_filter=[128])

    #conv4_x
    x = identity_block(x, nb_filter=[256], strides=(2, 2), with_conv_shortcut=True)
    x = identity_block(x, nb_filter=[256])
    x = identity_block(x, nb_filter=[256])
    x = identity_block(x, nb_filter=[256])
    x = identity_block(x, nb_filter=[256])
    x = identity_block(x, nb_filter=[256])

    #conv5_x
    x = identity_block(x, nb_filter=[512], strides=(2, 2), with_conv_shortcut=True)
    x = identity_block(x, nb_filter=[512])
    x = identity_block(x, nb_filter=[512])
    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model

def resnet_50(width,height,channel,classes):
    inpt = Input(shape=(width, height, channel))
    x = ZeroPadding2D((3, 3))(inpt)
    x = Conv2d_bN(x, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    #conv2_x
    x = bottleneck_block(x, nb_filters=[64,64,256],strides=(1,1),with_conv_shortcut=True)
    x = bottleneck_block(x, nb_filters=[64,64,256])
    x = bottleneck_block(x, nb_filters=[64,64,256])

    #conv3_x
    x = bottleneck_block(x, nb_filters=[128, 128, 512],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_block(x, nb_filters=[128, 128, 512])
    x = bottleneck_block(x, nb_filters=[128, 128, 512])
    x = bottleneck_block(x, nb_filters=[128, 128, 512])

    #conv4_x
    x = bottleneck_block(x, nb_filters=[256, 256, 1024],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_block(x, nb_filters=[256, 256, 1024])

    #conv5_x
    x = bottleneck_block(x, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_block(x, nb_filters=[512, 512, 2048])
    x = bottleneck_block(x, nb_filters=[512, 512, 2048])

    x = AveragePooling2D(pool_size=(7, 7))(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    return model


def acc_top2(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)


def check_print():
    input_shape = (48, 48, 1)
    nclasses = 7
    # Create a Keras Model
    model = resnet("resnet18", input_shape, nclasses)
    # model = resnet_50(input_shape, nclasses)
    model.summary()
    # Save a PNG of the Model Build
    plot_model(model, to_file='resnet.png')
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc', top_k_categorical_accuracy])
    print('Model Compiled')
    return model


if __name__ == '__main__':
    model = check_print()

 # loss, acc, top_acc = model.evaluate_generator(test_generator, steps=test_generator.n / batch_size)