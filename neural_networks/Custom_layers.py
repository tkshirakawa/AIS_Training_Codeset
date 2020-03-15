'''
    Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com

    Released under the BSD 3-Clause License
'''


##### For TensorFlow v2.0 #####
# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.layers import Layer
# from tensorflow.keras.initializers import Constant, Ones
# from tensorflow.keras.engine.base_layer import InputSpec

from keras import backend as K
from keras.layers import Layer
from keras.initializers import Constant, Ones
from keras.engine.base_layer import InputSpec




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

    def build(self, input_shape):
        self.alpha = self.add_weight(name        = 'alpha',
                                     shape       = list(input_shape[1:]),
                                     initializer = Ones(),
                                     trainable   = True)
        self.input_spec = InputSpec(ndim=len(input_shape))
        super(ParametricSwish, self).build(input_shape)

    def call(self, x):
        return K.sigmoid(self.alpha * x) * x

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

    def build(self, input_shape):
        self.alpha = self.add_weight(name        = 'alpha',
                                     shape       = list(input_shape[1:]),
                                     initializer = Constant(value=0.2),
                                     trainable   = True)
        self.input_spec = InputSpec(ndim=len(input_shape))
        super(FullSizePReLU, self).build(input_shape)

    def call(self, x):
        return K.relu(x) - self.alpha * K.relu(-x)

    def compute_output_shape(self, input_shape):
        return input_shape




'''
    Custom layer to calculates multiplying shift of input tensor.
    Note: keras.layers.Lambda() for a custom layer is NOT supported in Apple's coremltools for conversion to CoreML model.
'''
class MultiplyShift(Layer):

    def __init__(self, weight_init=1.0, **kwargs):
        super(MultiplyShift, self).__init__(**kwargs)
        self.weight_init = weight_init
        self.supports_masking = True

    def build(self, input_shape):
        self.weight = self.add_weight(name        = 'weight',
                                      shape       = list(input_shape[1:]),
                                      initializer = Constant(value=self.weight_init),
                                      trainable   = True)
        self.input_spec = InputSpec(ndim=len(input_shape))
        super(MultiplyShift, self).build(input_shape)

    def call(self, x):
        return self.weight * x

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'weight_init': self.weight_init}
        base_config = super(MultiplyShift, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




'''
    Custom layer to calculates learnable K.sigmoid of input tensor.
    Note: keras.layers.Lambda() for a custom layer is NOT supported in Apple's coremltools for conversion to CoreML model.
'''
class SigmoidShift(Layer):

    def __init__(self, alpha_init=1.0, translate_x=0.0, **kwargs):
        super(SigmoidShift, self).__init__(**kwargs)
        self.alpha_init = alpha_init
        self.translate_x = translate_x
        self.supports_masking = True

    def build(self, input_shape):
        self.alpha = self.add_weight(name         = 'alpha',
                                     shape        = list(input_shape[1:]),
                                     initializer  = Constant(value=self.alpha_init),
                                     trainable    = True)
        self.input_spec = InputSpec(ndim=len(input_shape))
        super(SigmoidShift, self).build(input_shape)

    def call(self, x):
        return K.sigmoid(self.alpha * (x - self.translate_x))

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {'alpha_init': self.alpha_init, 'translate_x': self.translate_x}
        base_config = super(SigmoidShift, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))




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



