"""MobileNet v2 models for Keras.

MobileNetV2 is a general architecture and can be used for multiple use cases.
Depending on the use case, it can use different input layer size and
different width factors. This allows different width models to reduce
the number of multiply-adds and thereby
reduce inference cost on mobile devices.

MobileNetV2 is very similar to the original MobileNet,
except that it uses inverted residual blocks with
bottlenecking features. It has a drastically lower
parameter count than the original MobileNet.
MobileNets support any input size greater
than 32 x 32, with larger image sizes
offering better performance.

The number of parameters and number of multiply-adds
can be modified by using the `alpha` parameter,
which increases/decreases the number of filters in each layer.
By altering the image size and `alpha` parameter,
all 22 models from the paper can be built, with ImageNet weights provided.

The paper demonstrates the performance of MobileNets using `alpha` values of
1.0 (also called 100 % MobileNet), 0.35, 0.5, 0.75, 1.0, 1.3, and 1.4

For each of these `alpha` values, weights for 5 different input image sizes
are provided (224, 192, 160, 128, and 96).


The following table describes the performance of
MobileNet on various input sizes:
------------------------------------------------------------------------
MACs stands for Multiply Adds

 Classification Checkpoint| MACs (M) | Parameters (M)| Top 1 Accuracy| Top 5 Accuracy
--------------------------|------------|---------------|---------|----|-------------
| [mobilenet_v2_1.4_224]  | 582 | 6.06 |          75.0 | 92.5 |
| [mobilenet_v2_1.3_224]  | 509 | 5.34 |          74.4 | 92.1 |
| [mobilenet_v2_1.0_224]  | 300 | 3.47 |          71.8 | 91.0 |
| [mobilenet_v2_1.0_192]  | 221 | 3.47 |          70.7 | 90.1 |
| [mobilenet_v2_1.0_160]  | 154 | 3.47 |          68.8 | 89.0 |
| [mobilenet_v2_1.0_128]  | 99  | 3.47 |          65.3 | 86.9 |
| [mobilenet_v2_1.0_96]   | 56  | 3.47 |          60.3 | 83.2 |
| [mobilenet_v2_0.75_224] | 209 | 2.61 |          69.8 | 89.6 |
| [mobilenet_v2_0.75_192] | 153 | 2.61 |          68.7 | 88.9 |
| [mobilenet_v2_0.75_160] | 107 | 2.61 |          66.4 | 87.3 |
| [mobilenet_v2_0.75_128] | 69  | 2.61 |          63.2 | 85.3 |
| [mobilenet_v2_0.75_96]  | 39  | 2.61 |          58.8 | 81.6 |
| [mobilenet_v2_0.5_224]  | 97  | 1.95 |          65.4 | 86.4 |
| [mobilenet_v2_0.5_192]  | 71  | 1.95 |          63.9 | 85.4 |
| [mobilenet_v2_0.5_160]  | 50  | 1.95 |          61.0 | 83.2 |
| [mobilenet_v2_0.5_128]  | 32  | 1.95 |          57.7 | 80.8 |
| [mobilenet_v2_0.5_96]   | 18  | 1.95 |          51.2 | 75.8 |
| [mobilenet_v2_0.35_224] | 59  | 1.66 |          60.3 | 82.9 |
| [mobilenet_v2_0.35_192] | 43  | 1.66 |          58.2 | 81.2 |
| [mobilenet_v2_0.35_160] | 30  | 1.66 |          55.7 | 79.1 |
| [mobilenet_v2_0.35_128] | 20  | 1.66 |          50.8 | 75.0 |
| [mobilenet_v2_0.35_96]  | 11  | 1.66 |          45.5 | 70.4 |

The weights for all 16 models are obtained and
translated from the Tensorflow checkpoints
from TensorFlow checkpoints found [here]
(https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md).

# Reference

This file contains building code for MobileNetV2, based on
[MobileNetV2: Inverted Residuals and Linear Bottlenecks]
(https://arxiv.org/abs/1801.04381)

Tests comparing this model to the existing Tensorflow model can be
found at [mobilenet_v2_keras]
(https://github.com/JonathanCMitchell/mobilenet_v2_keras)
"""
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, ZeroPadding2D
from tensorflow.keras.layers import Activation, BatchNormalization, add, Reshape, ReLU
from tensorflow.keras.layers import DepthwiseConv2D, Conv2D
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K


class MobileNetV2(object):
    def __init__(self,
                 input_shape,
                 alpha,
                 depth_multiplier=1,
                 classes=1000,
                 dropout=0.1
                 ):
        self.alpha = alpha
        self.input_shape = input_shape
        self.depth_multiplier = depth_multiplier
        self.classes = classes
        self.dropout = dropout

    @property
    def _first_conv_filters_number(self):
        return self._make_divisiable(32 * self.alpha, 8)

    @staticmethod
    def _make_divisiable(v, divisor=8, min_value=None):
        """
           This function is taken from the original tf repo.
           It ensures that all layers have a channel number that is divisible by 8
           It can be seen here:
           https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
        """
        if min_value is None:
            min_value = divisor
        new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
        # Make sure that round down does not go down by more than 10%.
        if new_v < 0.9 * v:
            new_v += divisor
        return new_v

    # 现在都不用池化层，而是采用stride=2来降采样
    @staticmethod
    def _correct_pad(x_input, kernel_size):
        """Returns a tuple for zero-padding for 2D convolution with downsampling.

        Args:
            x_input: An integer or tuple/list of 2 integers.
            kernel_size: An integer or tuple/list of 2 integers.

        Returns:
            A tuple.

        """
        img_dim = 1
        # 获取张量的shape,取出
        input_size = K.int_shape(x_input)[img_dim: img_dim + 2]
        # 检查输入是单个数字还是元祖,并且进行参数检查
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if input_size[0] is None:
            adjust = (1, 1)
        else:
            # % 取余数， // 取整
            adjust = (1 - input_size[0] % 2, 1 - input_size[1] % 2)
        correct = (kernel_size[0] // 2, kernel_size[1] // 2)

        return ((correct[0] - adjust[0], correct[0]),
                (correct[1] - adjust[1], correct[1]))

    def _first_conv_block(self, x_input):
        with K.name_scope('first_conv_block'):
            # 对2D输入（如图片）的边界填充0，以控制卷积以后特征图的大小
            x = ZeroPadding2D(padding=self._correct_pad(x_input, kernel_size=(3, 3)),
                              name='Conv1_pad')(x_input)
            x = Conv2D(filters=self._first_conv_filters_number,
                       kernel_size=3,
                       strides=(2, 2),
                       padding='valid',
                       use_bias=False,
                       name='Conv1')(x)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Bn_Conv1')(x)
            x = ReLU(max_value=6, name='Conv1_Relu')(x)
        return x

    def _inverted_res_block(self, x_input, filters, alpha, stride, expansion=1, block_id=0):
        """inverted residual block.

        Args:
            x_input: Tensor, Input tensor
            filters: the original filters of projected
            alpha: controls the width of the network. width multiplier.
            stride: the stride of depthwise convolution
            expansion: expand factor
            block_id: ID

        Returns:
            A tensor.
        """
        in_channels = K.int_shape(x_input)[-1]
        x = x_input
        prefix = 'block_{}_'.format(block_id)

        with K.name_scope("inverted_res_" + prefix):
            with K.name_scope("expand_block"):
                # 1. 利用 1x1 卷积扩张 从 filters--> expandsion x filters
                if block_id:  # 0为False，其余均为True
                    expandsion_channels = expansion * in_channels  # 扩张卷积的数量
                    x = Conv2D(filters=expandsion_channels,
                               kernel_size=(1, 1),
                               padding='same',
                               use_bias=False,
                               name=prefix + 'expand_Conv')(x)
                    x = BatchNormalization(epsilon=1e-3,
                                           momentum=0.999,
                                           name=prefix + 'expand_BN')(x)
                    x = ReLU(max_value=6, name=prefix + 'expand_Relu')(x)
                else:
                    prefix = 'expanded_conv_'

            with K.name_scope("depthwise_block"):
                # 2. Depthwise
                # 池化类型
                if stride == 2:
                    x = ZeroPadding2D(padding=self._correct_pad(x, (3, 3)),
                                      name=prefix + 'pad')(x)
                _padding = 'same' if stride == 1 else 'valid'
                x = DepthwiseConv2D(kernel_size=(3, 3),
                                    strides=stride,
                                    use_bias=False,
                                    padding=_padding,
                                    name=prefix + 'depthwise_Conv')(x)
                x = BatchNormalization(epsilon=1e-3,
                                       momentum=0.999,
                                       name=prefix + 'depthwise_Relu')(x)

            with K.name_scope("prpject_block"):
                # 3. Projected back to low-dimensional
                # 缩减的数量,output shape = _make_divisiable(int(filters * alpha))
                pointwise_conv_filters = self._make_divisiable(int(filters * alpha))
                x = Conv2D(filters=pointwise_conv_filters,
                           kernel_size=(1, 1),
                           padding='same',
                           use_bias=False,
                           name=prefix + 'project_Conv')(x)
                x = BatchNormalization(epsilon=1e-3,
                                       momentum=0.999,
                                       name=prefix +
                                            'project_BN')(x)
            # 4. shortcut
            if in_channels == pointwise_conv_filters and stride == 1:
                # alpha=1,stride=1,这样才能够使用使用shortcut
                x = add([x_input, x], name=prefix + 'add')
        return x

    def _last_conv_block(self, x_input):
        with K.name_scope("last_conv_block"):
            if self.alpha > 1.0:
                last_block_filters_number = self._make_divisiable(1280 * self.alpha, 8)
            else:
                last_block_filters_number = 1280

            x = Conv2D(last_block_filters_number,
                       kernel_size=(1, 1),
                       use_bias=False,
                       name='Conv_last')(x_input)
            x = BatchNormalization(epsilon=1e-3, momentum=0.999, name='Conv_last_bn')(x)
            return ReLU(max_value=6, name='out_relu')(x)

    def _conv_replace_dense(self, x_input):
        # 用卷积替代Dense
        # shape变为了x_input的通道数，即前一层的filters数量
        with K.name_scope('conv_dense'):
            x = GlobalAveragePooling2D()(x_input)
            x = Reshape(target_shape=(1, 1, 1280), name='reshape_1')(x)
            x = Dropout(self.dropout, name='dropout')(x)
            x = Conv2D(filters=self.classes, kernel_size=(1, 1),
                       padding='same', name='convolution')(x)
            x = Activation('softmax')(x)
            x = Reshape(target_shape=(self.classes,), name='reshape_2')(x)
        return x

    def block_bone(self):
        x_input = Input(shape=self.input_shape, name='Input')

        x = self._first_conv_block(x_input=x_input)

        x = self._inverted_res_block(x_input=x, filters=16, alpha=self.alpha,
                                     stride=1, expansion=1, block_id=0)

        x = self._inverted_res_block(x, 24, alpha=self.alpha, stride=2,
                                     expansion=6, block_id=1)
        x = self._inverted_res_block(x, 24, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=2)

        x = self._inverted_res_block(x, 32, alpha=self.alpha, stride=2,
                                     expansion=6, block_id=3)
        x = self._inverted_res_block(x, 32, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=4)
        x = self._inverted_res_block(x, 32, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=5)

        x = self._inverted_res_block(x, 64, alpha=self.alpha, stride=2,
                                     expansion=6, block_id=6)
        x = self._inverted_res_block(x, 64, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=7)
        x = self._inverted_res_block(x, 64, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=8)
        x = self._inverted_res_block(x, 64, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=9)

        x = self._inverted_res_block(x, 96, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=10)
        x = self._inverted_res_block(x, 96, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=11)
        x = self._inverted_res_block(x, 96, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=12)

        x = self._inverted_res_block(x, 160, alpha=self.alpha, stride=2,
                                     expansion=6, block_id=13)
        x = self._inverted_res_block(x, 160, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=14)
        x = self._inverted_res_block(x, 160, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=15)

        x = self._inverted_res_block(x, 320, alpha=self.alpha, stride=1,
                                     expansion=6, block_id=16)

        x = self._last_conv_block(x)
        x = self._conv_replace_dense(x)

        return Model(inputs=x_input, outputs=x)


# if __name__ == '__main__':
#     Mobilenet_V2_Model = MobileNetV2(input_shape=(224, 224, 3), alpha=1, classes=1000).block_bone()
#     Mobilenet_V2_Model.summary()
    # plot_model(Mobilenet_V2_Model, show_shapes=True, to_file='mobilenetv2.png')

#
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.layers import Activation, BatchNormalization, Add, Reshape
# from keras.applications.mobilenet import relu6, DepthwiseConv2D
#
# # from tensorflow.keras.utils import plot_model
#
#
# def _make_divisible(v, divisor, min_value=None):
#     if min_value is None:
#         min_value = divisor
#     new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
#     # Make sure that round down does not go down by more than 10%.
#     if new_v < 0.9 * v:
#         new_v += divisor
#     return new_v
#
#
# # def relu6(x):
# #     """Relu 6
# #     """
# #     return ReLU(max_value=6.0)(x)
#
#
# def _conv_block(inputs, filters, kernel, strides, stage):
#     channel_axis = 3
#     x = Conv2D(filters, kernel, padding='same', strides=strides, name='Conv'+str(stage))(inputs)
#     x = BatchNormalization(axis=channel_axis, name='bn_Conv'+str(stage))(x)
#     return Activation(relu6)(x)
#
#
# def _bottleneck(inputs, filters, kernel, t, alpha, s, r=False):
#     channel_axis = 3
#     # Depth
#     tchannel = tf.keras.backend.int_shape(inputs)[channel_axis] * t
#     # Width
#     cchannel = int(filters * alpha)
#
#     x = _conv_block(inputs, tchannel, (1, 1), (1, 1))
#
#     x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
#     x = BatchNormalization(axis=channel_axis)(x)
#     x = Activation(relu6)(x)
#
#     x = Conv2D(cchannel, (1, 1), strides=(1, 1), padding='same')(x)
#     x = BatchNormalization(axis=channel_axis)(x)
#
#     if r:
#         x = Add()([x, inputs])
#
#     return x
#
#
# def _inverted_residual_block(inputs, filters, kernel, t, alpha, strides, n):
#     x = _bottleneck(inputs, filters, kernel, t, alpha, strides)
#
#     for i in range(1, n):
#         x = _bottleneck(x, filters, kernel, t, alpha, 1, True)
#
#     return x
#
#
# def MobileNetv2(input_shape, num_classes, alpha=1.0, weights_path=None):
#
#     inputs = Input(shape=input_shape)
#
#     first_filters = _make_divisible(32 * alpha, 8)
#     x = _conv_block(inputs, first_filters, (3, 3), strides=(2, 2), stage=1)
#
#     x = _inverted_residual_block(x, 16, (3, 3), t=1, alpha=alpha, strides=1, n=1)
#     x = _inverted_residual_block(x, 24, (3, 3), t=6, alpha=alpha, strides=2, n=2)
#     x = _inverted_residual_block(x, 32, (3, 3), t=6, alpha=alpha, strides=2, n=3)
#     x = _inverted_residual_block(x, 64, (3, 3), t=6, alpha=alpha, strides=2, n=4)
#     x = _inverted_residual_block(x, 96, (3, 3), t=6, alpha=alpha, strides=1, n=3)
#     x = _inverted_residual_block(x, 160, (3, 3), t=6, alpha=alpha, strides=2, n=3)
#     x = _inverted_residual_block(x, 320, (3, 3), t=6, alpha=alpha, strides=1, n=1)
#
#     if alpha > 1.0:
#         last_filters = _make_divisible(1280 * alpha, 8)
#     else:
#         last_filters = 1280
#
#     x = _conv_block(x, last_filters, (1, 1), strides=(1, 1))
#     x = GlobalAveragePooling2D()(x)
#     x = Reshape((1, 1, last_filters))(x)
#     x = Dropout(0.3, name='Dropout')(x)
#     x = Conv2D(num_classes, (1, 1), padding='same')(x)
#
#     x = Activation('softmax', name='softmax')(x)
#     output = Reshape((num_classes,))(x)
#
#     model = Model(inputs, output)
#     if weights_path:
#         model.load_weights(weights_path, by_name=True)
#     # plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)
#
#     return model
#
#
# if __name__ == '__main__':
#     model = MobileNetv2((224, 224, 3), 100, 1.0)
#     print(model.summary())

#
# import tensorflow as tf
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
# from tensorflow.keras.layers import Activation, BatchNormalization, add, Reshape
# from tensorflow.keras.applications.mobilenet import relu6, DepthwiseConv2D
# from tensorflow.keras.utils.vis_utils import plot_model
#
#
# def _conv_block(inputs, filters, kernel, strides):
#     channel_axis = 3
#     x = Conv2D(filters, kernel, padding='same', strides=strides)(inputs)
#     x = BatchNormalization(axis=channel_axis)(x)
#     return Activation(relu6)(x)
#
#
# def _bottleneck(inputs, filters, kernel, t, s, r=False):
#     channel_axis = 3
#     tchannel = tf.keras.backend.int_shape(inputs)[channel_axis] * t
#
#     x = _conv_block(inputs, tchannel, (1, 1), (1, 1))
#
#     x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same')(x)
#     x = BatchNormalization(axis=channel_axis)(x)
#     x = Activation(relu6)(x)
#
#     x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same')(x)
#     x = BatchNormalization(axis=channel_axis)(x)
#     if r:
#         x = add([x, inputs])
#         return x
#
#
# def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
#     x = _bottleneck(inputs, filters, kernel, t, strides)
#     for i in range(1, n):
#         x = _bottleneck(x, filters, kernel, t, 1, True)
#         return x
#
# # input_shape, num_classes, weights_path=None, pooling=None
# def MobileNetv2(input_shape, num_classes, weights_path=None):
#     inputs = Input(shape=input_shape)
#     x = _conv_block(inputs, 32, (3, 3), strides=(2, 2))
#
#     x = _inverted_residual_block(x, 16, (3, 3), t=1, strides=1, n=1)
#     x = _inverted_residual_block(x, 24, (3, 3), t=6, strides=2, n=2)
#     x = _inverted_residual_block(x, 32, (3, 3), t=6, strides=2, n=3)
#     x = _inverted_residual_block(x, 64, (3, 3), t=6, strides=2, n=4)
#     x = _inverted_residual_block(x, 96, (3, 3), t=6, strides=1, n=3)
#     x = _inverted_residual_block(x, 160, (3, 3), t=6, strides=2, n=3)
#     x = _inverted_residual_block(x, 320, (3, 3), t=6, strides=1, n=1)
#
#     x = _conv_block(x, 1280, (1, 1), strides=(1, 1))
#     x = GlobalAveragePooling2D()(x)
#     x = Reshape((1, 1, 1280))(x)
#     x = Dropout(0.3, name='Dropout')(x)
#     x = Conv2D(num_classes, (1, 1), padding='same')(x)
#
#     x = Activation('softmax', name='softmax')(x)
#     output = Reshape((num_classes,))(x)
#
#     model = Model(inputs, output)
#     if weights_path:
#         model.load_weights(weights_path, by_name=True)
#
#     plot_model(model, to_file='images/MobileNetv2.png', show_shapes=True)
#     return model
#
#
# if __name__ == '__main__':
#     MobileNetv2((224, 224, 3), 1000)
