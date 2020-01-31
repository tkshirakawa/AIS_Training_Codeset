'''
    <<< About this network >>>
    U-net implementation for A.I.Segmentation
    Revised from the following original.

    - Original codes -
    URL: https://github.com/chuckyee/cardiac-segmentation
    URL: https://blog.insightdatascience.com/heart-disease-diagnosis-with-deep-learning-c2d92c27e730

    - Original copyrights -
    Copyright (c) 2017 chuckyee
    Released under the MIT license
    URL: https://github.com/chuckyee/cardiac-segmentation/blob/master/LICENSE
'''




from keras.models import Model
from keras.layers import Input, BatchNormalization, Activation, Dropout
from keras.layers.convolutional import Conv2D, Conv2DTranspose, UpSampling2D, ZeroPadding2D, Cropping2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import Concatenate
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.initializers import Constant
from keras import backend as K




####################################################################################################
####    Descriptions and definitions
####################################################################################################

def Model_Name(): return 'U-net'


def Model_Description(): return 'U-net implementation for A.I.Segmentation\n\
                            Revised from the following original.\n\
                            - Original codes -\n\
                            URL: https://github.com/chuckyee/cardiac-segmentation\n\
                            URL: https://blog.insightdatascience.com/heart-disease-diagnosis-with-deep-learning-c2d92c27e730\n\
                            - Original copyrights -\n\
                            Copyright (c) 2017 chuckyee\n\
                            Released under the MIT license\n\
                            URL: https://github.com/chuckyee/cardiac-segmentation/blob/master/LICENSE'


'''
    Dictionary of custom layers used in the following Build_Model().
    This will be used for conversion to CoreML model.
    Use the same name for keys and values in this dictionary.

    If you use custom layers, "return { 'custom_layer1_name': custom_layer1_def, 'custom_layer2_name': custom_layer2_def, ... }".
    
    If you do not use custom layers, just "return {}".
'''
def Custom_Layers(): return {}


'''
    Define a batch size used for training.
'''
def Batch_Size(): return 8


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


'''
    Activation layers.
'''
# def ActivationBy(activation='relu', alpha=0.2):
    
#     if activation == 'leakyrelu':
#         return LeakyReLU(alpha=alpha)

#     elif activation == 'prelu':
#         return PReLU(alpha_initializer=Constant(alpha), shared_axes=[1, 2])

#     elif activation == 'elu':
#         return ELU(alpha=alpha)

#     else:
#         return Activation(activation)




def downsampling_block(input_tensor, filters, padding='valid', batchnorm=False, dropout=0.0):

    _, height, width, _ = K.int_shape(input_tensor)
    assert height % 2 == 0
    assert width % 2 == 0

    x = Conv2D(filters, kernel_size=(3,3), padding=padding)(input_tensor)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    return MaxPooling2D(pool_size=(2,2))(x), x




def upsampling_block(input_tensor, skip_tensor, filters, padding='valid', batchnorm=False, dropout=0.0):

    x = Conv2DTranspose(filters, kernel_size=(2,2), strides=(2,2))(input_tensor)

    # compute amount of cropping needed for skip_tensor
    _, x_height, x_width, _ = K.int_shape(x)
    _, s_height, s_width, _ = K.int_shape(skip_tensor)
    h_crop = s_height - x_height
    w_crop = s_width - x_width
    assert h_crop >= 0
    assert w_crop >= 0
    if h_crop == 0 and w_crop == 0:
        y = skip_tensor
    else:
        cropping = ((h_crop//2, h_crop - h_crop//2), (w_crop//2, w_crop - w_crop//2))
        y = Cropping2D(cropping=cropping)(skip_tensor)

    x = Concatenate()([x, y])

    x = Conv2D(filters, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    return x




def unet(features=64, depth=4, padding='valid', batchnorm=False, dropout=0.0):
    
    """Generate U-Net model introduced in
      "U-Net: Convolutional Networks for Biomedical Image Segmentation"
      O. Ronneberger, P. Fischer, T. Brox (2015)
    Arbitrary number of input channels and output classes are supported.
    Arguments:
      height  - input image height (pixels)
      width   - input image width  (pixels)
      channels - input image features (1 for grayscale, 3 for RGB)
      classes - number of output classes (2 in paper)
      features - number of output features for first convolution (64 in paper)
          Number of features double after each down sampling block
      depth  - number of downsampling operations (4 in paper)
      padding - 'valid' (used in paper) or 'same'
      batchnorm - include batch normalization layers before activations
      dropout - fraction of units to dropout, 0 to keep all units
    Output:
      U-Net model expecting input shape (height, width, maps) and generates
      output with shape (output_height, output_width, classes). If padding is
      'same', then output_height = height and output_width = width.
    """

    # x = Input(shape=(height, width, channels))
    # inputs = x
    inputs = Input(shape=(200, 200, 1), name='input')
    x = ZeroPadding2D(4)(inputs)

    skips = []
    for i in range(depth):
        x, x0 = downsampling_block(x, features, padding, batchnorm, dropout)
        skips.append(x0)
        features *= 2

    x = Conv2D(filters=features, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    x = Conv2D(filters=features, kernel_size=(3,3), padding=padding)(x)
    x = BatchNormalization()(x) if batchnorm else x
    x = Activation('relu')(x)
    x = Dropout(dropout)(x) if dropout > 0 else x

    for i in reversed(range(depth)):
        features //= 2
        x = upsampling_block(x, skips[i], features, padding, batchnorm, dropout)

    # x = Conv2D(filters=classes, kernel_size=(1,1))(x)
    x = Conv2D(filters=1, kernel_size=(1,1), activation='sigmoid')(x)
    outputs = Cropping2D(4, name='output')(x)

    # logits = Lambda(lambda z: z/temperature)(x)
    # probabilities = Activation('softmax')(logits)

    return Model(inputs=inputs, outputs=outputs)




def Build_Model():
    return unet(features=64, depth=4, padding='same', batchnorm=True, dropout=0.2)

