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
from keras.layers import Input, Activation, Dropout
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, Conv2DTranspose, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate, Maximum, Average, Add, Multiply
# from keras.layers.noise import GaussianDropout, AlphaDropout  : Not supported in coremltools, Mar 18, 2019
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.initializers import Constant

# Trial for 16-bit float calculation
from keras.layers import BatchNormalization as BNLayer
# from F16_func.BNF16 import BatchNormalizationF16 as BNLayer



def Model_Name():
    return 'CV-net'



def Model_Description():
    return         'Neural network model for A.I.Segmentation\n\
                    Constructed for segmentation of cardiovascular CT images\n\
                    Copyright (c) 2019-2020, Takashi Shirakawa\n\
                    URL: https://compositecreatures.jimdofree.com/a-i-segmentation/'



def ActivationBy(activation='relu', alpha=0.2):

    if activation == 'leakyrelu':
        return LeakyReLU(alpha=alpha)
    
    elif activation == 'prelu':
        return PReLU(alpha_initializer=Constant(alpha), shared_axes=[1, 2])
    
    elif activation == 'elu':
        return ELU(alpha=alpha)
    
    else:
        return Activation(activation)



def Build_Model():

    # OpenCV(grayscale) = HEIGHT x WIDTH
    # Keras = HEIGHT x WIDTH x CHANNEL
    inputs = Input(shape=(200, 200, 1), name='input')

    
    # C1
    c1 = Conv2D(filters=16, kernel_size=(7, 7), padding='same', kernel_initializer='he_uniform')(inputs)
    c1 = DepthwiseConv2D(kernel_size=(3, 3), padding='same', depthwise_initializer='he_normal', use_bias=False)(c1)
    c1 = BNLayer(axis=-1, momentum=0.9, epsilon=0.001)(c1)
    c1 = ActivationBy('prelu', alpha=0.2)(c1)
    #c1 = ActivationBy('leakyrelu', alpha=0.1)(c1)

    
    # C2
    c2 = Conv2D(filters=64, kernel_size=(24, 24), padding='valid', kernel_initializer='he_uniform')(c1)
    c2 = DepthwiseConv2D(kernel_size=(5, 5), padding='valid', depthwise_initializer='he_normal', use_bias=False)(c2)
    c2 = BNLayer(axis=-1, momentum=0.9, epsilon=0.001)(c2)
    c2 = ActivationBy('prelu', alpha=0.1)(c2)
    c2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='valid')(c2)

    # C3
    c3 = Conv2D(filters=128, kernel_size=(1, 1), padding='valid', kernel_initializer='he_uniform')(c2)
    c3 = DepthwiseConv2D(kernel_size=(3, 3), padding='valid', depthwise_initializer='he_normal', use_bias=False)(c3)
    c3 = BNLayer(axis=-1, momentum=0.9, epsilon=0.001)(c3)
    c3 = ActivationBy('prelu', alpha=0.1)(c3)


    # AveragePooling C3
    avp = AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c3)
    #avp = Dropout(0.25)(avp)
    #avp = GaussianDropout(0.1)(avp)

    # C41
    c41 = Conv2D(filters=256, kernel_size=(5, 5), padding='valid', kernel_initializer='he_uniform')(avp)
    c41 = DepthwiseConv2D(kernel_size=(3, 3), padding='valid', depthwise_initializer='he_normal', use_bias=False)(c41)
    c41 = BNLayer(axis=-1, momentum=0.9, epsilon=0.001)(c41)
    c41 = ActivationBy('prelu', alpha=0.1)(c41)
    
    # C42
    c42 = Conv2D(filters=256, kernel_size=(1, 1), padding='valid', kernel_initializer='he_uniform')(avp)
    c42 = DepthwiseConv2D(kernel_size=(7, 7), padding='valid', depthwise_initializer='he_normal', use_bias=False)(c42)
    c42 = BNLayer(axis=-1, momentum=0.9, epsilon=0.001)(c42)
    c42 = ActivationBy('prelu', alpha=0.1)(c42)
    #c42 = ActivationBy('elu', alpha=1.0)(c42)
    
    
    # MaxPooling C3
    mxp = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(c3)
    mxp = Dropout(0.1)(mxp)
    #mxp = AlphaDropout(0.1)(mxp)
    
    # C43
    c43 = Conv2D(filters=256, kernel_size=(23, 23), padding='valid', kernel_initializer='he_uniform')(mxp)
    c43 = DepthwiseConv2D(kernel_size=(3, 3), padding='valid', depthwise_initializer='he_normal', use_bias=False)(c43)
    c43 = BNLayer(axis=-1, momentum=0.9, epsilon=0.001)(c43)
    c43 = ActivationBy('relu')(c43)
    #c43 = ActivationBy('prelu', alpha=0.2)(c43)
    c43 = UpSampling2D(size=(2, 2))(c43)


    # Max
    #c41 = Add()([c41, c41])
    #c4  = Add()([c42, c43])
    #c4 = Maximum()([c41, c4])
    c4 = Maximum()([c41, c42, c43])


    # D1a
    d1a = Conv2DTranspose(filters=128, kernel_size=(13, 13), padding='valid', kernel_initializer='he_uniform', use_bias=False)(c4)
    d1a = BNLayer(axis=-1, momentum=0.9, epsilon=0.001)(d1a)
    d1a = ActivationBy('prelu', alpha=0.1)(d1a)
    d1a = UpSampling2D(size=(2, 2))(d1a)

    # D2a
    d2a = Conv2DTranspose(filters=32, kernel_size=(5, 5), padding='valid', kernel_initializer='he_uniform', use_bias=False)(d1a)
    d2a = BNLayer(axis=-1, momentum=0.9, epsilon=0.001)(d2a)
    d2a = ActivationBy('prelu', alpha=0.1)(d2a)
    d2a = UpSampling2D(size=(2, 2))(d2a)


    # D1b
    d1b = Conv2DTranspose(filters=64, kernel_size=(5, 5), padding='valid', kernel_initializer='he_uniform', use_bias=False)(c4)
    d1b = BNLayer(axis=-1, momentum=0.9, epsilon=0.001)(d1b)
    d1b = ActivationBy('prelu', alpha=0.1)(d1b)
    d1b = UpSampling2D(size=(2, 2))(d1b)
    
    # D2b
    d2b = Conv2DTranspose(filters=16, kernel_size=(21, 21), padding='valid', kernel_initializer='he_uniform', use_bias=False)(d1b)
    d2b = BNLayer(axis=-1, momentum=0.9, epsilon=0.001)(d2b)
    d2b = ActivationBy('prelu', alpha=0.1)(d2b)
    d2b = UpSampling2D(size=(2, 2))(d2b)

    d2b = Multiply()([c1, d2b])


    # Concatenate
    d2 = Concatenate(axis=-1)([c1, d2a, d2b])

    # O
    od = Conv2D(filters=16, kernel_size=(1, 1), padding='same', kernel_initializer='he_uniform')(d2)
    od = DepthwiseConv2D(kernel_size=(5, 5), padding='same', depthwise_initializer='he_normal', use_bias=False)(od)
    od = BNLayer(axis=-1, momentum=0.9, epsilon=0.001)(od)
    od = ActivationBy('leakyrelu', alpha=0.01)(od)
    od = Add()([od, od, od, od])

    od = Conv2D(filters=1, kernel_size=(1, 1), padding='valid', kernel_initializer='he_uniform')(od)
    outputs = Activation('sigmoid', name='output')(od)


    # Generate model
    return Model(inputs=inputs, outputs=outputs)


