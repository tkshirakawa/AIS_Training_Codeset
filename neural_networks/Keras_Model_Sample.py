

from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout, SpatialDropout2D, Layer
from keras.layers.convolutional import Conv2D, DepthwiseConv2D, Conv2DTranspose, UpSampling2D, Cropping2D, ZeroPadding2D
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.merge import Concatenate, Maximum, Average, Add, Multiply
from keras.layers.advanced_activations import ReLU, LeakyReLU, PReLU, ELU
from keras.initializers import Constant
from keras import backend as K

# If you want to use custom layers
# The search path for Custom_layers must be the path from Train.py,
# because this NN model file is called in Train.py.
from neural_networks.Custom_layers import Swish




####################################################################################################
####    Descriptions and definitions
####################################################################################################

def Model_Name(): return 'NN_AIS_sample'


def Model_Description(): return 'Neural network model for A.I.Segmentation (SAMPLE)\n\
                            Constructed for organ segmentation from medical images\n\
                            Copyright (c) 2019-2020, Takashi Shirakawa\n\
                            URL: https://compositecreatures.jimdofree.com/a-i-segmentation/'


'''
    Dictionary of custom layers used in the following Build_Model().
    This will be used for conversion to CoreML model.
    Use the same name for keys and values in this dictionary.

    If you use custom layers, "return { 'custom_layer1_name': custom_layer1_def, 'custom_layer2_name': custom_layer2_def, ... }".
    
    If you do not use custom layers, just "return {}".
'''
def Custom_Layers(): return { 'Swish': Swish }      # return {}


'''
    Define a batch size used for training.
'''
def Batch_Size(): return 16


'''
    Define learning rates at the points of epochs : [[epoch, learning rate], ...].
    Learning rates between epochs will be interpolated linearly.
    The epoch value in the last component of this list is the total epoch number when training finishes.
'''
# def Learning_Rate_Lsit(): return [[0,3e-3], [2,3e-3]]   # TEST
# def Learning_Rate_Lsit(): return [[0,3e-3], [20,3e-3], [50,2e-3], [80,5e-4]]
# def Learning_Rate_Lsit(): return [[0,3e-3], [10,3e-3], [50,1e-3], [70,5e-4], [100,2e-4]]
# def Learning_Rate_Lsit(): return [[0,3e-3], [10,3e-3], [50,1e-3], [70,5e-4], [100,2e-4]]
def Learning_Rate_Lsit(): return [[0, 5e-3], [5, 1.5e-2], [10, 1.5e-2], [20, 1e-2], [30, 7.5e-3], [50, 5e-3]]
# def Learning_Rate_Lsit(): return [[0,3e-3], [3,3.2e-3], [12,4.5e-3], [30,4.5e-3], [50,3e-3], [80,1e-3], [100,5e-4], [150,2e-4]]




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
    outputs = Activation('sigmoid', name='output')(cx)


    # Generate model
    return Model(inputs=inputs, outputs=outputs)


