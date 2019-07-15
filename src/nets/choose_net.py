#!/usr/bin/env python
# encoding: utf-8
"""
@Author: JianboZhu
@Contact: jianbozhu1996@gmail.com
@Date: 2019/5/30
@Description:
"""
import os
from .vgg import vgg16
# from .mobilenet import MobileNetV2
from .mobilenet_v2 import MobileNetV2
pretrain_vgg16_model_path = os.path.join('..', 'trained_models', 'pretrain_models',
                                         "vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5")
pretrain_mobilenet_model_path = os.path.join('..', 'trained_models', 'pretrain_models',
                                             "mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5")
# mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5
# mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_128_no_top.h5

def choose_net(name, input_shape, num_classes):
    if name == 'VGG16':
        if os.path.exists(pretrain_vgg16_model_path):
            model = vgg16(input_shape, num_classes, pretrain_vgg16_model_path)
            print_info(True, model, pretrain_vgg16_model_path)
        else:
            model = vgg16(input_shape, num_classes)
            print_info(False, model, pretrain_vgg16_model_path)
    elif name == "MobileNet":
        if os.path.exists(pretrain_mobilenet_model_path):
            model = MobileNetV2(input_shape, num_classes, pretrain_mobilenet_model_path)
            print_info(True, model, pretrain_mobilenet_model_path)
        else:
            model = MobileNetV2(input_shape, num_classes).block_bone()
            print_info(False, model, pretrain_mobilenet_model_path)

    elif name == "Resnet34":
        # TODO
        pass
    else:
        raise Exception("Only VGG16 or MobileNet now!")

    return model


def print_info(issuccess, model, info):
    model.summary()
    if issuccess:
        print("\n**************************************************************")
        print(" Successful load: vgg16 model does not exist in path\n {}!".format(info))
        print("**************************************************************\n")
    else:
        # raise Exception("{} does not exist!".format(pretrain_vgg16_model_path))
        print("\n**************************************************************")
        print(" WARNING: vgg16 model does not exist in path\n {}!".format(info))
        print(" Did not load pretrain model!")
        print("**************************************************************\n")
