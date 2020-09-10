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
from keras.layers import Input, BatchNormalization, Activation, Dropout, SpatialDropout2D, Layer
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, Conv2DTranspose, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate, Maximum, Average, Add, Multiply
from keras.layers.advanced_activations import ReLU, LeakyReLU, PReLU, ELU
from keras.initializers import Constant




####################################################################################################
####    Descriptions and definitions
####################################################################################################

def Model_Name(): return 'NN_AIS_sample'


def Model_Description(): return 'Neural network model for A.I.Segmentation (SAMPLE)\n\
                            Constructed for organ segmentation from medical images\n\
                            Copyright (c) 2019-2020, Takashi Shirakawa\n\
                            URL: https://compositecreatures.jimdofree.com/a-i-segmentation/'


'''
    Define the number of classes.
'''
def Number_of_Classes(): return 1


'''
    Dictionary of custom layers used in the following Build_Model().
    This will be used for conversion to CoreML model.
    Use the same name for keys and values in this dictionary.

    If you use custom layers, "return { 'custom_layer1_name': custom_layer1_def, 'custom_layer2_name': custom_layer2_def, ... }".
    
    If you do not use custom layers, just "return {}".
'''
# If you want to use custom layers
# The search path for Custom_layers must be the path from Train.py,
# because this NN model file is called in Train.py.
from neural_networks.Custom_layers import Swish, ParametricSwish, FullSizePReLU
def Custom_Layers(): return { 'Swish': Swish }      # return {}


'''
    Define a batch size used for training.
'''
def Batch_Size(): return 16


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
def Learning_Rate_Formula(): return ['poly', 0.50, 100]         # For AS calcium

def Learning_Rate_Lsit(): return [[0, 5e-3], [5, 1.5e-2], [10, 1.5e-2], [20, 1e-2], [30, 7.5e-3], [50, 5e-3]]


'''
    Define a count number before early stopping.
'''
def Count_before_Stop(): return 20




####################################################################################################
####    Main neural network
####################################################################################################

def ActivationBy(function='relu', value=0.2, max_value=None, threshold=0.0):

    # Custom activation layers
    if function == 'swish':
        return Swish()

    # Default activation layers
    elif function == 'relu':
        return ReLU(max_value=max_value, negative_slope=value, threshold=threshold)

    elif function == 'leakyrelu':
        return LeakyReLU(alpha=value)
    
    elif function == 'prelu':
        return PReLU(alpha_initializer=Constant(value=value), shared_axes=[1, 2])
    
    elif function == 'elu':
        return ELU(alpha=value)
    
    else:
        return Activation(function)




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


    # Do not change "name='input'", because the name is used to identify the input layer in A.I.Segmentation.
    inputs = Input(shape=(200, 200, 1), name='input')


    #  Sample
    c1 = Conv2D(filters=16, kernel_size=(3, 3), padding='same')(inputs)
    c1 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001)(c1)
    c1 = ActivationBy(function='relu')(c1)


    # Sample
    c2 = Conv2D(filters=64, kernel_size=(3, 3), padding='same')(c1)
    c2 = BatchNormalization(axis=-1, momentum=0.9, epsilon=0.001)(c2)
    c2 = ActivationBy(function='swish')(c2)


    #
    # Your layers...
    #
    cx = ...


    # Output
    # Do not change "name='output'", because the name is used to identify the output layer in A.I.Segmentation.
    outputs = Conv2D(filters=_num_classes, kernel_size=1, kernel_initializer='glorot_uniform')(cx)
    outputs = Activation('sigmoid', name='output')(outputs)


    # Generate model
    return Model(inputs=inputs, outputs=outputs)



