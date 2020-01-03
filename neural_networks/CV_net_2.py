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
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, Conv2DTranspose, UpSampling2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate, Add
from keras import backend as K




def Model_Name():
    return 'CV-net2'




def Model_Description():
    return 'Neural network model for A.I.Segmentation\n\
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




# Dictionary of custom layers used in the following Build_Model() for conversion to CoreML model
# Use the same name for keys and values of the following dictionary
def Custom_Layers():
    return { 'Swish': Swish }




def Build_Model():

    # OpenCV(grayscale) = HEIGHT x WIDTH
    # Keras = HEIGHT x WIDTH x CHANNEL
    inputs = Input(shape=(200, 200, 1), name='input')


    # Encoder 0
    enc0 = BatchNormalization(momentum=0.9)(inputs)
    enc0 = Swish()(enc0)
    enc0 = Concatenate()([inputs, enc0])


    # Encoder 1
    enc1 = Conv2D(filters=16, kernel_size=7, padding='same', kernel_initializer='he_uniform')(enc0)
    enc1 = DepthwiseConv2D(kernel_size=3, padding='same', depthwise_initializer='he_uniform', use_bias=False)(enc1)
    enc1 = BatchNormalization(momentum=0.9)(enc1)
    enc1 = Swish()(enc1)


    # Encoder 2
    enc2 = Conv2D(filters=64, kernel_size=24, padding='valid', kernel_initializer='he_uniform')(enc1)
    enc2 = DepthwiseConv2D(kernel_size=5, padding='valid', depthwise_initializer='he_uniform', use_bias=False)(enc2)
    enc2 = BatchNormalization(momentum=0.9)(enc2)
    enc2 = Swish()(enc2)
    enc2 = MaxPooling2D(pool_size=3, strides=2, padding='valid')(enc2)


    # Encoder 3
    enc3 = Conv2D(filters=128, kernel_size=1, padding='valid', kernel_initializer='he_uniform')(enc2)
    enc3 = DepthwiseConv2D(kernel_size=3, padding='valid', depthwise_initializer='he_uniform', use_bias=False)(enc3)
    enc3 = BatchNormalization(momentum=0.9)(enc3)
    enc3 = Swish()(enc3)
    enc3 = MaxPooling2D(pool_size=2, strides=2, padding='valid')(enc3)


    # Encoder 4 - Main
    enc4M = Dropout(0.2)(enc3)
    enc4M = Conv2D(filters=384, kernel_size=23, padding='valid', kernel_initializer='he_uniform')(enc4M)
    enc4M = DepthwiseConv2D(kernel_size=3, padding='valid', depthwise_initializer='he_uniform', use_bias=False)(enc4M)
    enc4M = BatchNormalization(momentum=0.9)(enc4M)
    enc4M = Swish()(enc4M)
    enc4M = UpSampling2D(size=2)(enc4M)


    # Encoder 4 - Line A
    enc4A = Conv2D(filters=192, kernel_size=5, padding='valid', kernel_initializer='he_uniform')(enc3)
    enc4A = DepthwiseConv2D(kernel_size=3, padding='valid', depthwise_initializer='he_uniform', use_bias=False)(enc4A)
    enc4A = BatchNormalization(momentum=0.9)(enc4A)
    enc4A = Swish()(enc4A)
    
    # Encoder 4 - Line B
    enc4B = Conv2D(filters=192, kernel_size=1, padding='valid', kernel_initializer='he_uniform')(enc3)
    enc4B = DepthwiseConv2D(kernel_size=7, padding='valid', depthwise_initializer='he_uniform', use_bias=False)(enc4B)
    enc4B = BatchNormalization(momentum=0.9)(enc4B)
    enc4B = Swish()(enc4B)


    # Encoder to decoder
    # Cross connection with dropout and add - Line A
    enc2dec_A = Concatenate()([enc4A, SpatialDropout2D(0.2)(enc4B)])
    enc2dec_A = Add()([enc2dec_A, enc4M])

    # Cross connection with dropout and add - Line B
    enc2dec_B = Concatenate()([enc4B, SpatialDropout2D(0.2)(enc4A)])
    enc2dec_B = Add()([enc2dec_B, enc4M])


    # Decoder 1 - Line A
    dec1A = Conv2DTranspose(filters=128, kernel_size=13, padding='valid', kernel_initializer='he_uniform', use_bias=False)(enc2dec_A)
    dec1A = BatchNormalization(momentum=0.9)(dec1A)
    dec1A = Swish()(dec1A)
    dec1A = UpSampling2D(size=2)(dec1A)

    # Decoder 2 - Line A
    dec2A = Conv2DTranspose(filters=48, kernel_size=5, padding='valid', kernel_initializer='he_uniform', use_bias=False)(dec1A)
    dec2A = BatchNormalization(momentum=0.9)(dec2A)
    dec2A = Swish()(dec2A)
    dec2A = UpSampling2D(size=2)(dec2A)


    # Decoder 1 - Line B
    dec1B = Conv2DTranspose(filters=128, kernel_size=5, padding='valid', kernel_initializer='he_uniform', use_bias=False)(enc2dec_B)
    dec1B = BatchNormalization(momentum=0.9)(dec1B)
    dec1B = Swish()(dec1B)
    dec1B = UpSampling2D(size=2)(dec1B)
    
    # Decoder 2 - Line B
    dec2B = Conv2DTranspose(filters=48, kernel_size=21, padding='valid', kernel_initializer='he_uniform', use_bias=False)(dec1B)
    dec2B = BatchNormalization(momentum=0.9)(dec2B)
    dec2B = Swish()(dec2B)
    dec2B = UpSampling2D(size=2)(dec2B)


    # Concatenate
    # conc_enc1_dec2 = Concatenate()([enc1, dec2A, dec2B])
    conc_enc1_dec2_A = Concatenate()([dec2A, SpatialDropout2D(0.2)(dec2B), enc1])
    conc_enc1_dec2_B = Concatenate()([dec2B, SpatialDropout2D(0.2)(dec2A), enc1])


    # Decoder 3 - Line A
    dec3A = Conv2D(filters=16, kernel_size=3, padding='same', kernel_initializer='he_uniform')(conc_enc1_dec2_A)
    dec3A = DepthwiseConv2D(kernel_size=3, padding='same', depthwise_initializer='he_uniform', use_bias=False)(dec3A)
    dec3A = BatchNormalization(momentum=0.9)(dec3A)
    dec3A = Swish()(dec3A)


    # Decoder 3 - Line B
    dec3B = Conv2D(filters=16, kernel_size=1, padding='same', kernel_initializer='he_uniform')(conc_enc1_dec2_B)
    dec3B = DepthwiseConv2D(kernel_size=5, padding='same', depthwise_initializer='he_uniform', use_bias=False)(dec3B)
    dec3B = BatchNormalization(momentum=0.9)(dec3B)
    dec3B = Swish()(dec3B)


    # Output
    # outputs = Concatenate()([enc0, dec3])
    outputs = Add()([dec3A, dec3B])
    outputs = Conv2D(filters=1, kernel_size=1, padding='valid', kernel_initializer='he_uniform')(outputs)
    outputs = Activation('sigmoid', name='output')(outputs)


    # Generate model
    return Model(inputs=inputs, outputs=outputs)


