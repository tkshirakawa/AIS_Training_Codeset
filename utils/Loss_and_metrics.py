'''
    Copyright (c) 2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com

    Released under the BSD 3-Clause License

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
'''




##### For TensorFlow v2.0 #####
# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.losses import mean_squared_error, mean_absolute_error, hinge, squared_hinge, logcosh

import tensorflow as tf
from keras import backend as K
from keras.losses import mean_squared_error, mean_absolute_error, hinge, squared_hinge, logcosh




def mean_squared_error_loss(y_true, y_pred): return mean_squared_error(y_true, y_pred)

def mean_absolute_error_loss(y_true, y_pred): return mean_absolute_error(y_true, y_pred)


def dilation(tensor, size=5): return K.pool2d(tensor, pool_size=(size,size), strides=(1,1), padding="same", pool_mode='max')

def erosion(tensor, size=5): return -K.pool2d(-tensor, pool_size=(size,size), strides=(1,1), padding="same", pool_mode='max')

def morphology_gradient(tensor, size=5):  return dilation(tensor, size=size) - erosion(tensor, size=size)

def sum2d(tensor): return K.sum(tensor)
    # if K.image_data_format == 'channels_first':  return K.sum(tensor, axis=[2,3])         # batch_size(N), ch(C), row(H), col(W)
    # elif K.image_data_format == 'channels_last': return K.sum(tensor, axis=[1,2])         # batch_size(N), row(H), col(W), ch(C)


mean_axis = None
# mean_axis = -1




####################################################################################################
######      IoU
####################################################################################################

def mean_iou(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1.)       # (batch_size, row, col, ch)
    y_pred = K.clip(y_pred, K.epsilon(), 1.)
    intersection = sum2d(y_true * y_pred)
    union = sum2d(y_true) + sum2d(y_pred) - intersection
    return K.mean(intersection / union, axis=mean_axis)

# def mean_iou(y_true, y_pred):
#     y_true = K.cast_to_floatx(y_true >= 0.5, tf.int32)
#     y_pred = K.cast_to_floatx(y_pred >= 0.5, tf.int32)
#     mean_iou, _ = tf.compat.v1.metrics.mean_iou(y_true, y_pred, num_classes=1)
#     return  tf.reduce_mean(mean_iou)

# MIOU = tf.keras.metrics.MeanIoU(num_classes=1, dtype=tf.float32)
# def mean_iou(y_true, y_pred):
#     y_true = K.cast_to_floatx(y_true >= 0.5, tf.int32)
#     y_pred = K.cast_to_floatx(y_pred >= 0.5, tf.int32)
#     MIOU.update_state(y_true, y_pred)
#     return  MIOU.result()


def mean_sq_iou(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1.)       # (batch_size, row, col, ch)
    y_pred = K.clip(y_pred, K.epsilon(), 1.)
    intersection = sum2d(y_true * y_pred)
    union = sum2d(y_true) + sum2d(y_pred) - intersection
    return K.mean(K.square(intersection / union), axis=mean_axis)


def mean_sigmoid_iou(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1.)       # (batch_size, row, col, ch)
    y_pred = K.clip(y_pred, K.epsilon(), 1.)
    intersection = sum2d(y_true * y_pred)
    union = sum2d(y_true) + sum2d(y_pred) - intersection
    return K.mean(K.sigmoid(6.0 * (intersection / union - 0.5)), axis=mean_axis)


def mean_iou_loss(y_true, y_pred):
    return 1. - mean_iou(y_true, y_pred)

### range = 0 - 2
def mean_iou_MSE_loss(y_true, y_pred):
    return mean_iou_loss(y_true, y_pred) + mean_squared_error(y_true, y_pred)

###
def mean_iou_MSE2_loss(y_true, y_pred):
    return 0.5 * mean_iou_loss(y_true, y_pred) + 1.5 * mean_squared_error(y_true, y_pred)

###
def mean_iou_edged_MSE_loss(y_true, y_pred):
    y_true_mol = morphology_gradient(y_true, size=11)
    y_pred_mol = morphology_gradient(y_pred, size=11)
    return mean_iou_loss(y_true, y_pred) + mean_squared_error(y_true, y_pred) + mean_squared_error(y_true_mol, y_pred_mol)


def mean_sq_iou_MSE_loss(y_true, y_pred):
    return 1. - mean_sq_iou(y_true, y_pred) + mean_squared_error(y_true, y_pred)


def mean_sigmoid_iou_MSE_loss(y_true, y_pred):
    return 1. - mean_sigmoid_iou(y_true, y_pred) + mean_squared_error(y_true, y_pred)


def mean_sigmoid_iou_edged_MSE_loss(y_true, y_pred):
    y_true_mol = morphology_gradient(y_true, size=11)
    y_pred_mol = morphology_gradient(y_pred, size=11)
    return mean_sigmoid_iou_MSE_loss(y_true, y_pred) + mean_squared_error(y_true_mol, y_pred_mol)


def mean_iou_MAE_loss(y_true, y_pred):
    return mean_iou_loss(y_true, y_pred) + mean_absolute_error(y_true, y_pred)


def mean_iou_SVM_loss(y_true, y_pred):
    return mean_iou_loss(y_true, y_pred) + hinge(y_true, y_pred)


def mean_iou_SSVM_loss(y_true, y_pred):
    return mean_iou_loss(y_true, y_pred) + squared_hinge(y_true, y_pred)


def mean_iou_LGC_loss(y_true, y_pred):
    return mean_iou_loss(y_true, y_pred) + logcosh(y_true, y_pred)




####################################################################################################
######      RoU: residual over union
####################################################################################################

# RoU: residual over union
# The residual is uncovered residual area in grand-truth. RoU = grand-truth - intersection
def mean_rou(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1.)       # (batch_size, row, col, ch)
    y_pred = K.clip(y_pred, K.epsilon(), 1.)
    true_sum = sum2d(y_true)
    intersection = sum2d(y_true * y_pred)
    union = true_sum + sum2d(y_pred) - intersection
    return K.mean((true_sum - intersection) / union, axis=mean_axis)
    # return (true_sum - intersection + smooth) / (true_sum + sum2d(y_pred) - intersection + smooth)


# mean_iou_rou = (mean_iou + (1 - mean_rou)) / 2    range: 0 - 1
def mean_iou_rou(y_true, y_pred):
    return 0.5 * (mean_iou(y_true, y_pred) + (1. - mean_rou(y_true, y_pred)))


def mean_iou_rou_MSE_loss(y_true, y_pred):
    return 1. - mean_iou_rou(y_true, y_pred) + mean_squared_error(y_true, y_pred)




####################################################################################################
######      Huber loss
####################################################################################################

def Huber_loss(y_true, y_pred):
    return tf.compat.v1.losses.huber_loss(y_true, y_pred, weights=25.0, delta=0.5)


def mean_iou_Huber_loss(y_true, y_pred):
    return mean_iou_loss(y_true, y_pred) + Huber_loss(y_true, y_pred)




####################################################################################################
######      IoU of/with morphology gradient
####################################################################################################

def mean_moliou(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1.)       # (batch_size, row, col, ch)
    y_pred = K.clip(y_pred, K.epsilon(), 1.)
    y_true_mol = morphology_gradient(y_true, size=11)
    y_pred_mol = morphology_gradient(y_pred, size=11)
    return mean_iou(y_true_mol, y_pred_mol)


def mean_moliou_loss(y_true, y_pred):
    return 1. - mean_moliou(y_true, y_pred)


def mean_moliou_MSE_loss(y_true, y_pred):
    return mean_moliou_loss(y_true, y_pred) + mean_squared_error(y_true, y_pred)




####################################################################################################
######      Dice
####################################################################################################

def dice_coef(y_true, y_pred):
    y_true = K.clip(y_true, K.epsilon(), 1.)       # (batch_size, row, col, ch)
    y_pred = K.clip(y_pred, K.epsilon(), 1.)
    numerator = 2. * sum2d(y_true * y_pred)
    denominator = sum2d(y_true) + sum2d(y_pred)
    return K.mean(numerator / denominator, axis=mean_axis)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def dice_coef_MSE_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + mean_squared_error(y_true, y_pred)


def dice_coef_MAE_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + mean_absolute_error(y_true, y_pred)


def dice_coef_SVM_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + hinge(y_true, y_pred)


def dice_coef_SSVM_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + squared_hinge(y_true, y_pred)


def dice_coef_LGC_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + logcosh(y_true, y_pred)


