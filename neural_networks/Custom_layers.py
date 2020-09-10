'''
    Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com
    
    
    Released under the MIT license.
    https://opensource.org/licenses/mit-license.php

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


import math

import tensorflow as tf

from keras import backend as K
from keras.layers import Layer
from keras.initializers import Constant, Ones, TruncatedNormal
from keras.constraints import NonNeg
from keras.engine.base_layer import InputSpec

##### For TensorFlow v2 #####
# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Layer, InputSpec
# from tensorflow.keras.initializers import Constant, Ones, TruncatedNormal
# from tensorflow.keras.constraints import NonNeg




'''
    Knowledge source : Custom Layers in Core ML written by Matthijs Hollemans.
    https://machinethink.net/blog/coreml-custom-layers/
'''
class Swish(Layer):

    def __init__(self, **kwargs):
        super(Swish, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Swish, self).build(input_shape)

    def call(self, x):
        return K.sigmoid(x) * x

    def compute_output_shape(self, input_shape):
        return input_shape




'''
    Swish with a learnable parameter.
    Knowledge source : Custom Layers in Core ML written by Matthijs Hollemans.
    https://machinethink.net/blog/coreml-custom-layers/
'''
class ParametricSwish(Layer):

    def __init__(self, **kwargs):
        super(ParametricSwish, self).__init__(**kwargs)
        self.supports_masking = True

    # input_shape = batch_size(N), ch(C), row(H), col(W) or batch_size(N), row(H), col(W), ch(C)
    def build(self, input_shape):
        stddev = 0.0
        if K.image_data_format() == 'channels_first':  stddev = math.sqrt(2.0 / max(float(input_shape[1]), 32.0))
        elif K.image_data_format() == 'channels_last': stddev = math.sqrt(2.0 / max(float(input_shape[-1]), 32.0))
        self.alpha = self.add_weight(name        = 'alpha',
                                     shape       = list(input_shape[1:]),
                                     initializer = TruncatedNormal(mean=1.0, stddev=stddev),
                                     trainable   = True,
                                     constraint  = NonNeg() )
        self.input_spec = InputSpec(ndim=len(input_shape))
        super(ParametricSwish, self).build(input_shape)

    def call(self, x):
        beta = 1.5 - 0.5 ** self.alpha
        return K.sigmoid(beta * x) * x

    def compute_output_shape(self, input_shape):
        return input_shape




'''
    PReLU without shared axes.
    Because PReLU(shared_axes=None) causes error 'shared_axes must be [1,2] or [1,2,3]' in coremltools,
    this custom PReLU is required for PReLU(shared_axes=None).
'''
class FullSizePReLU(Layer):

    def __init__(self, **kwargs):
        super(FullSizePReLU, self).__init__(**kwargs)
        self.supports_masking = True

    # input_shape = batch_size(N), ch(C), row(H), col(W) or batch_size(N), row(H), col(W), ch(C)
    def build(self, input_shape):
        stddev = 0.0
        if K.image_data_format() == 'channels_first':  stddev = math.sqrt(2.0 / max(float(input_shape[1]), 32.0))
        elif K.image_data_format() == 'channels_last': stddev = math.sqrt(2.0 / max(float(input_shape[-1]), 32.0))
        self.alpha = self.add_weight(name        = 'alpha',
                                     shape       = list(input_shape[1:]),
                                     initializer = TruncatedNormal(mean=0.2, stddev=stddev),
                                     trainable   = True)
        self.input_spec = InputSpec(ndim=len(input_shape))
        super(FullSizePReLU, self).build(input_shape)

    def call(self, x):
        return K.relu(x) - self.alpha * K.relu(-x)

    def compute_output_shape(self, input_shape):
        return input_shape




'''
    Custom layer to calculates K.square of input tensor.
    Note: keras.layers.Lambda() for a custom layer is NOT supported in Apple's coremltools for conversion to CoreML model.
'''
class Square(Layer):

    def __init__(self, **kwargs):
        super(Square, self).__init__(**kwargs)

    def build(self, input_shape):
        super(Square, self).build(input_shape)

    def call(self, x):
        return K.square(x)

    def compute_output_shape(self, input_shape):
        return input_shape




'''
    Custom layer to calculates K.sqrt of input tensor.
    Note: keras.layers.Lambda() for a custom layer is NOT supported in Apple's coremltools for conversion to CoreML model.
'''
class SQRT(Layer):

    def __init__(self, **kwargs):
        super(SQRT, self).__init__(**kwargs)

    def build(self, input_shape):
        super(SQRT, self).build(input_shape)

    def call(self, x):
        return K.sqrt(x)

    def compute_output_shape(self, input_shape):
        return input_shape



