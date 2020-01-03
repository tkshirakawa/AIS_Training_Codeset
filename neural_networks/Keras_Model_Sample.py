'''
    Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com
    
    
    Released under the BSD license.
    URL: https://opensource.org/licenses/BSD-2-Clause
    
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''




from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout, SpatialDropout2D, Layer
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, Conv2DTranspose, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate, Maximum, Average, Add, Multiply
from keras.layers.advanced_activations import ReLU, LeakyReLU, PReLU, ELU
from keras.initializers import Constant
from keras import backend as K




def Model_Name():
    return 'NN_AIS_sample'




def Model_Description():
    return 'Neural network model for A.I.Segmentation (SAMPLE)\n\
            Constructed for organ segmentation from medical images\n\
            Copyright (c) 2019-2020, Takashi Shirakawa\n\
            URL: https://compositecreatures.jimdofree.com/a-i-segmentation/'




# Implementation of a custom layer, Swish
# Knowledge source : Custom Layers in Core ML written by Matthijs Hollemans
# https://machinethink.net/blog/coreml-custom-layers/

class Swish(Layer):
    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Swish, self).build(input_shape)

    def call(self, x):
        return K.sigmoid(x) * x

    def compute_output_shape(self, input_shape):
        return input_shape




# For conversion to CoreML model
# Dictionary of custom layers used in the following Build_Model
# If you use custom layers, "return { 'custom_layer1_name': custom_layer1_def, 'custom_layer2_name': custom_layer2_def, ... }".
# If you do not use custom layers, just "return {}".
def Custom_Layers():
    return { 'Swish': Swish }
    # return {}




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




def Build_Model():

    # OpenCV(grayscale) = HEIGHT x WIDTH
    # Keras = HEIGHT x WIDTH x CHANNEL
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
    # Do not change "name='output'", because the name is used to identify the input layer in A.I.Segmentation.
    outputs = Activation('sigmoid', name='output')(cx)


    # Generate model
    return Model(inputs=inputs, outputs=outputs)


