'''
    Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com
    
    
    Released under the MIT license.
    https://opensource.org/licenses/mit-license.php

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


import tensorflow as tf

from keras import backend as K
from keras.models import Model
from keras.layers import Input, Layer, BatchNormalization, Activation, Dropout, SpatialDropout2D, ZeroPadding2D
from keras.layers.noise import GaussianDropout, GaussianNoise
from keras.layers.convolutional import Conv2D, SeparableConv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate

##### For TensorFlow v2 #####
# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Layer, BatchNormalization, Activation, ZeroPadding2D
# from tensorflow.keras.layers import Conv2D, SeparableConv2D, Conv2DTranspose, UpSampling2D, MaxPooling2D
# from tensorflow.keras.layers import Concatenate
# from tensorflow.keras.layers import Dropout, SpatialDropout2D, GaussianDropout, GaussianNoise




'''
    Synaptic Transmission Regulator, STR.
    The layer calculates and learn only two parameters, linear weight and bias for input tensor.
'''
import math

from keras.initializers import TruncatedNormal
from keras.regularizers import l2
from keras.engine.base_layer import InputSpec

##### For TensorFlow v2 #####
# from tensorflow.keras.initializers import TruncatedNormal
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras.layers import InputSpec

class SynapticTransmissionRegulator(Layer):

    def __init__(self, **kwargs):
        super(SynapticTransmissionRegulator, self).__init__(**kwargs)
        self.supports_masking = True

    # input_shape = batch_size(N), ch(C), row(H), col(W) or batch_size(N), row(H), col(W), ch(C)
    # The length of input_shape, len(input_shape), must be 4
    def build(self, input_shape):
        channel_axis, shape = 0, [1,1,1]
        if K.image_data_format() == 'channels_first':  channel_axis = 1     # 0:ch(C),  1:row(H), 2:col(W)
        elif K.image_data_format() == 'channels_last': channel_axis = 3     # 0:row(H), 1:col(W), 2:ch(C)
        unit_num = input_shape[channel_axis]
        shape[channel_axis-1] = unit_num
        stddev = math.sqrt(2.0 /  max(float(unit_num), 32.0))
        self.weight = self.add_weight(name        = 'weight',
                                      shape       = shape,
                                      initializer = TruncatedNormal(mean=1.2, stddev=stddev),
                                      regularizer = l2(1e-4),
                                      trainable   = True)
        self.bias   = self.add_weight(name        = 'bias',
                                      shape       = shape,
                                      initializer = TruncatedNormal(mean=0.0, stddev=stddev),
                                      regularizer = l2(1e-4),
                                      trainable   = True)
        self.input_spec = InputSpec(ndim=4, axes={channel_axis: unit_num})
        super(SynapticTransmissionRegulator, self).build(input_shape)

    def call(self, x):
        return self.weight * x + self.bias

    def compute_output_shape(self, input_shape):
        return input_shape




####################################################################################################
####    Descriptions and definitions
####################################################################################################

def Model_Name(): return 'CV-net SYNAPSE'


def Model_Description(): return 'Neural network model for A.I.Segmentation plugin\n\
                            CV-net SYNAPSE is the 3rd generation neural network of CV-net\n\
                            Constructed for segmentation of medical images\n\
                            Copyright (c) 2020, Takashi Shirakawa\n\
                            URL: https://compositecreatures.jimdofree.com/a-i-segmentation/'


'''
    Define the number of classes.
'''
def Number_of_Classes(): return 1


'''
    Dictionary of custom layers used in the following Build_Model().
    This will be used for conversion to CoreML model. Use the same name for keys and values in this dictionary.

    If you use custom layers, "return { 'custom_layer1_name': custom_layer1_def, 'custom_layer2_name': custom_layer2_def, ... }".
    If you do not use custom layers, just "return {}".
'''
# Import if you want to use custom layers
# Note: The search path for Custom_layers must be the path from Train.py, because this NN model file is called in Train.py.
# from neural_networks.Custom_layers import Swish, ParametricSwish, FullSizePReLU
from neural_networks.Custom_layers import ParametricSwish

# def Custom_Layers(): return { 'Swish': Swish, 'SynapticTransmissionRegulator': SynapticTransmissionRegulator }
def Custom_Layers(): return { 'ParametricSwish': ParametricSwish, 'SynapticTransmissionRegulator': SynapticTransmissionRegulator }


'''
    Define a batch size used for training.
'''
# def Batch_Size(): return 16     # for nVidia RTX2070, 8GB VRAM
def Batch_Size(): return 80     # for nVidia TITAN RTX, 24GB VRAM


'''
    Define an optimizer used for training.
'''
from keras.optimizers import SGD, Adam, Nadam
# from tensorflow.keras.optimizers import SGD, Adam, Nadam      ##### For TensorFlow v2 #####
def Optimizer(base_lr=0.1): return SGD(lr=base_lr, momentum=0.95, nesterov=True)
# def Optimizer(base_lr=0.001): return Adam(lr=base_lr, beta_1=0.9, beta_2=0.999, amsgrad=False)
# def Optimizer(base_lr=0.001): return Nadam(lr=base_lr, beta_1=0.9, beta_2=0.999)


'''
    Define learning rate
    By formula : ['poly', base_lr, number_of_epochs]
    By graph : [[epoch, learning rate], [..., ...] ..., [number_of_epochs, final learning rate]]
               Learning rates between epochs will be interpolated linearly.
    Note: The graph will NOT be used when the formula is defined. In other words, the formula is used prior to the graph.
'''
# def Learning_Rate_Formula(): return [None, 0.0, 0]
# def Learning_Rate_Formula(): return ['poly', 0.35, 80]          # For heart structures
def Learning_Rate_Formula(): return ['poly', 0.50, 100]         # For AS calcium
# def Learning_Rate_Formula(): return ['poly', 0.50, 50]
# def Learning_Rate_Formula(): return ['poly', 0.25, 50]        # For heart structures / SGD / batch size = 16
# def Learning_Rate_Formula(): return ['poly', 0.5, 50]         # For heart structures / SGD / batch size = 42
# def Learning_Rate_Formula(): return ['poly', 0.5, 100]        # For brain tumor / SGD / batch size = 16
# def Learning_Rate_Formula(): return ['poly', 0.25, 20]        # For aorta / SGD

# def Learning_Rate_Lsit(): return [[0,1e-3], [50,5e-1]]   # TEST
# def Learning_Rate_Lsit(): return [[0,3e-3], [10,3e-3], [50,1e-3], [70,5e-4], [100,2e-4]]
# def Learning_Rate_Lsit(): return [[0, 1e-3], [5, 1.5e-3], [15, 1e-3], [30, 1e-4]]     # Not good
# def Learning_Rate_Lsit(): return [[0, 5e-3], [5, 1e-2], [10, 1e-2], [20, 2e-3]]     # For aorta / Adam, Nadam

# LR used for comparing CV-net SYNAPSE, DeepLab v3+ and U-net
def Learning_Rate_Lsit(): return [[0, 5e-3], [5, 1.5e-2], [10, 1.5e-2], [20, 1e-2], [30, 7.5e-3], [50, 5e-3]]


'''
    Define a count number before early stopping.
'''
def Count_before_Stop(): return 50




####################################################################################################
####    Main neural network
####################################################################################################

def SynapticNeuronUnit(dendrites, filter_size, kernel_size, CRP, d_rate, use_STR):

    if CRP[1] == 'UpSampling': dendrites = UpSampling2D(interpolation='bilinear')(dendrites)


    # Synaptic Transmission Regulator, STR, calculates weight and bias for each channel of input tensor
    if use_STR: neuro_potential = SynapticTransmissionRegulator()(dendrites)
    else:       neuro_potential = dendrites


    # Main neural potential
    if CRP[0] == 'Normal':
        neuro_potential = Conv2D(   filters             = filter_size,
                                    kernel_size         = kernel_size,
                                    padding             = CRP[2],
                                    kernel_initializer  = 'he_uniform',
                                    use_bias            = False)(neuro_potential)

    elif CRP[0] == 'Transpose':
        neuro_potential = Conv2DTranspose(  filters             = filter_size,
                                            kernel_size         = kernel_size,
                                            padding             = CRP[2],
                                            kernel_initializer  = 'he_uniform',
                                            use_bias            = False)(neuro_potential)

    elif CRP[0] == 'Separable':
        neuro_potential = SeparableConv2D(  filters                 = filter_size,
                                            kernel_size             = kernel_size,
                                            padding                 = CRP[2],
                                            depthwise_initializer   = 'he_uniform',
                                            pointwise_initializer   = 'he_uniform',
                                            use_bias                = False)(neuro_potential)

    elif CRP[0] == 'Atrous':
        neuro_potential = Conv2D(   filters             = filter_size,
                                    kernel_size         = kernel_size,
                                    strides             = 2,
                                    padding             = CRP[2],
                                    kernel_initializer  = 'he_uniform',
                                    use_bias            = False)(neuro_potential)
        neuro_potential = ZeroPadding2D(padding=((1, 0), (1, 0)))(neuro_potential)

    else:
        neuro_potential = None      # Will be error

    neuro_potential = BatchNormalization(momentum=0.95)(neuro_potential)
    neuro_potential = ParametricSwish()(neuro_potential)


    # Output potential to axons
    if CRP[1] == 'MaxPooling': neuro_potential = MaxPooling2D()(neuro_potential)


    if d_rate[0] > 0.0: neuro_potential = GaussianDropout(rate=d_rate[0])(neuro_potential)
    if d_rate[1] > 0.0: neuro_potential = SpatialDropout2D(rate=d_rate[1])(neuro_potential)

    return neuro_potential




'''
    Build and return Keras model

    Input/output images are grayscale = 1 channel per pixel.
    Type of the pixels is float normalized between 0.0 to 1.0.
    (Please note that the pixel values are NOT 8-bit unsigned char ranging between 0 to 255)

    Dimensions
    OpenCV : HEIGHT x WIDTH
    Keras  : HEIGHT x WIDTH x CHANNEL
'''
def Build_Model():

    _num_classes = Number_of_Classes()          # Classes used in this NN

    # Convolution methods for SynapticNeuronUnit layers
    # _ENC = ('Normal', 'MaxPooling', 'same')
    _ENC = ('Atrous', 'None', 'valid')
    _NNv = ('Normal', 'None', 'valid')
    _SNs = ('Separable', 'None', 'same')
    _SNv = ('Separable', 'None', 'valid')
    _TUs = ('Transpose', 'UpSampling', 'same')

    # Rate for dropout and noise layers
    # NOTE: The larger values (around 0.5) may be better when segmentation will have large area such as the heart
    # On the other hand, for small segmentation area such as calcium deposits on a valve,
    # the rate for SpatialDropout2D should be ZERO and the stddev for GaussianNoise should be small
    _dropout_rate1  = (0.20, 0.0)           # (rate for GaussianDropout, rate for SpatialDropout2D)
    _dropout_rate2  = (0.40, 0.0)           # (rate for GaussianDropout, rate for SpatialDropout2D)
    _stddev_g_noise = 0.10                  # std.deviation for GaussianNoise


    # Do not change "name='input'", because the name is used to identify the input layer in A.I.Segmentation.
    inputs = Input(shape=(200, 200, 1), name='input')


    # Cerate stimulation from inputs
    stimulation = Conv2D(filters=8, kernel_size=9, kernel_initializer='glorot_uniform', use_bias=False)(inputs)
    stimulation = BatchNormalization(momentum=0.95)(stimulation)
    stimulation = Activation('sigmoid')(stimulation)                # Skip connection


    # Main neural network
    # def SynapticNeuronUnit(dendrites, filter_size, kernel_size, CRP, d_rate, use_STR)
    enc_potential_96  = SynapticNeuronUnit(stimulation,        16,  3, _ENC, _dropout_rate1, False)     # 96
    enc_potential_96A = SynapticNeuronUnit(enc_potential_96,   16,  1, _NNv, _dropout_rate2, False)
    enc_potential_96B = SynapticNeuronUnit(enc_potential_96,   16,  5, _SNs, _dropout_rate2, False)
    enc_potential_96  = Concatenate()([enc_potential_96A, enc_potential_96B])                               # Skip connection

    enc_potential_48  = SynapticNeuronUnit(enc_potential_96,   32,  3, _ENC, _dropout_rate1, False)     # 48
    enc_potential_48A = SynapticNeuronUnit(enc_potential_48,   32,  1, _NNv, _dropout_rate2, False)
    enc_potential_48B = SynapticNeuronUnit(enc_potential_48,   32,  5, _SNs, _dropout_rate2, False)
    enc_potential_48  = Concatenate()([enc_potential_48A, enc_potential_48B])                               # Skip connection

    enc_potential_24  = SynapticNeuronUnit(enc_potential_48,  128,  3, _ENC, _dropout_rate1, True)      # 24
    enc_potential_24A = SynapticNeuronUnit(enc_potential_24,  128,  1, _NNv, _dropout_rate2, False)
    enc_potential_24B = SynapticNeuronUnit(enc_potential_24,  128,  5, _SNs, _dropout_rate2, False)
    enc_potential_24  = Concatenate()([enc_potential_24A, enc_potential_24B])                               # Skip connection

    enc_potential_12  = SynapticNeuronUnit(enc_potential_24,  512,  3, _ENC, _dropout_rate1, True)      # 12
    enc_potential_12A = SynapticNeuronUnit(enc_potential_12,  512,  1, _NNv, _dropout_rate2, False)
    enc_potential_12B = SynapticNeuronUnit(enc_potential_12,  512,  5, _SNs, _dropout_rate2, False)
    enc_potential_12  = Concatenate()([enc_potential_12A, enc_potential_12B])                               # Skip connection

    deep_potential    = SynapticNeuronUnit(enc_potential_12, 1024,  5, _SNv, _dropout_rate1, True)      # 08
    deep_potential    = SynapticNeuronUnit(deep_potential,   2048,  3, _SNv, _dropout_rate1, True)      # 06

    dec_potential_12  = SynapticNeuronUnit(deep_potential,    512,  3, _TUs, _dropout_rate1, True)      # 12
    dec_potential_12  = Concatenate()([dec_potential_12, enc_potential_12])

    dec_potential_24  = SynapticNeuronUnit(dec_potential_12,  384,  3, _TUs, _dropout_rate1, True)      # 24
    dec_potential_24  = Concatenate()([dec_potential_24, enc_potential_24])

    dec_potential_48  = SynapticNeuronUnit(dec_potential_24,  160,  3, _TUs, _dropout_rate1, True)      # 48
    dec_potential_48  = Concatenate()([dec_potential_48, enc_potential_48])

    dec_potential_96  = SynapticNeuronUnit(dec_potential_48,   56,  3, _TUs, _dropout_rate1, False)     # 96
    dec_potential_96  = Concatenate()([dec_potential_96, enc_potential_96])

    axon_potential    = SynapticNeuronUnit(dec_potential_96,   24,  3, _TUs, _dropout_rate1, False)     # 192
    axon_potential    = Concatenate()([axon_potential, stimulation])


    # The vision from synaptic neurons
    vision = Conv2DTranspose(filters=32, kernel_size=7, kernel_initializer='he_uniform', use_bias=False)(axon_potential)
    vision = BatchNormalization(momentum=0.95)(vision)
    vision = ParametricSwish()(vision)
    vision = GaussianNoise(stddev=_stddev_g_noise)(vision)

    vision = Conv2DTranspose(filters=16, kernel_size=3, kernel_initializer='he_uniform', use_bias=False)(vision)
    vision = BatchNormalization(momentum=0.95)(vision)
    vision = ParametricSwish()(vision)
    vision = GaussianNoise(stddev=_stddev_g_noise)(vision)

    vision = Concatenate()([vision, inputs])


    # Do not change "name='output'", because the name is used to identify the output layer in A.I.Segmentation.
    outputs = Conv2D(filters=_num_classes, kernel_size=1, kernel_initializer='glorot_uniform')(vision)
    outputs = SynapticTransmissionRegulator()(outputs)
    outputs = Activation('sigmoid', name='output')(outputs)


    # Generate model
    return Model(inputs=inputs, outputs=outputs)



