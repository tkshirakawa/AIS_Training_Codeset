'''
    Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com

    Released under the BSD 3-Clause License
'''


##### For TensorFlow v2.0 #####
# import tensorflow as tf
# from tensorflow.keras import backend as K
# from tensorflow.keras.losses import mean_squared_error, mean_absolute_error, hinge, squared_hinge, logcosh

import tensorflow as tf
from keras import backend as K
from keras.losses import mean_squared_error, hinge, squared_hinge, logcosh, binary_crossentropy, kullback_leibler_divergence




num_classes = -1
def set_num_classes(n=1):
    global num_classes
    num_classes = n




def mean_squared_error_loss(y_true, y_pred): return mean_squared_error(y_true, y_pred)

def mean_absolute_error_loss(y_true, y_pred): return mean_absolute_error(y_true, y_pred)


def dilation(tensor, size=5): return K.pool2d(tensor, pool_size=(size,size), strides=(1,1), padding="same", pool_mode='max')

def erosion(tensor, size=5): return -K.pool2d(-tensor, pool_size=(size,size), strides=(1,1), padding="same", pool_mode='max')

def morphology_gradient(tensor, size=5): return dilation(tensor, size=size) - erosion(tensor, size=size)


def proc_tensors(y_true, y_pred):
    if not K.is_tensor(y_pred):
        y_pred = K.constant(y_pred)
    # y_true = K.clip(K.cast(y_true >= 0.5, y_pred.dtype), K.epsilon(), 1.)
    # y_pred = K.clip(K.cast(y_pred >= 0.5, y_pred.dtype), K.epsilon(), 1.)
    y_true = K.clip(y_true, K.epsilon(), 1.)
    y_pred = K.clip(y_pred, K.epsilon(), 1.)
    ch = -100
    if num_classes == 1:
        ch = None
    elif K.ndim(y_pred) == 4:
        if K.image_data_format() == 'channels_first':  ch = 1       # batch_size(N), ch(C), row(H), col(W)
        elif K.image_data_format() == 'channels_last': ch = -1      # batch_size(N), row(H), col(W), ch(C)
    return y_true, y_pred, ch




####################################################################################################
######      IoU
####################################################################################################

def mean_iou(y_true, y_pred):
    y_true, y_pred, ch = proc_tensors(y_true, y_pred)
    intersection = K.sum(K.sqrt(y_true * y_pred), axis=ch)      # y_XXXX is a kind of 'area'. So, the multiplied value should be reverted to its root.
    union = K.sum(y_true, axis=ch) + K.sum(y_pred, axis=ch) - intersection
    return K.mean(intersection / union, axis=ch)


def mean_iou_loss(y_true, y_pred):
    return 1. - mean_iou(y_true, y_pred)


def mean_iou_MSE_loss(y_true, y_pred):
    return mean_iou_loss(y_true, y_pred) + mean_squared_error(y_true, y_pred)


def mean_iou_MSE2_loss(y_true, y_pred):
    return 0.5 * mean_iou_loss(y_true, y_pred) + 1.5 * mean_squared_error(y_true, y_pred)


def mean_iou_BCE_loss(y_true, y_pred):
    return mean_iou_loss(y_true, y_pred) + binary_crossentropy(y_true, y_pred)


def mean_iou_MSE_border_MSE_loss(y_true, y_pred):
    y_true, y_pred, _ = proc_tensors(y_true, y_pred)
    y_true_mol = morphology_gradient(y_true, size=11)
    y_pred_mol = morphology_gradient(y_pred, size=11)
    return mean_iou_loss(y_true, y_pred) + mean_squared_error(y_true, y_pred) + mean_squared_error(y_true_mol, y_pred_mol)


def mean_iou_MSE_border_BCE_loss(y_true, y_pred):
    y_true, y_pred, _ = proc_tensors(y_true, y_pred)
    y_true_mol = morphology_gradient(y_true, size=11)
    y_pred_mol = morphology_gradient(y_pred, size=11)
    return mean_iou_loss(y_true, y_pred) + mean_squared_error(y_true, y_pred) + binary_crossentropy(y_true_mol, y_pred_mol)


def mean_iou_MSE_border_KLD_loss(y_true, y_pred):
    y_true, y_pred, _ = proc_tensors(y_true, y_pred)
    y_true_mol = morphology_gradient(y_true, size=11)
    y_pred_mol = morphology_gradient(y_pred, size=11)
    return mean_iou_loss(y_true, y_pred) + mean_squared_error(y_true, y_pred) + kullback_leibler_divergence(y_true_mol, y_pred_mol)


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
    y_true, y_pred, ch = proc_tensors(y_true, y_pred)
    true_sum = K.sum(y_true, axis=ch)
    intersection = K.sum(K.sqrt(y_true * y_pred), axis=ch)      # y_XXXX is a kind of 'area'. So, the multiplied value should be reverted to its root.
    union = true_sum + K.sum(y_pred, axis=ch) - intersection
    return K.mean((true_sum - intersection) / union, axis=ch)


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
######      Dice
####################################################################################################

def dice_coef(y_true, y_pred):
    y_true, y_pred, ch = proc_tensors(y_true, y_pred)
    numerator = 2. * K.sum(K.sqrt(y_true * y_pred), axis=ch)        # y_XXXX is a kind of 'area'. So, the multiplied value should be reverted to its root.
    denominator = K.sum(y_true, axis=ch) + K.sum(y_pred, axis=ch)
    return K.mean(numerator / denominator, axis=ch)


def dice_coef_loss(y_true, y_pred):
    return 1. - dice_coef(y_true, y_pred)


def dice_coef_MSE_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + mean_squared_error(y_true, y_pred)


def dice_coef_SVM_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + hinge(y_true, y_pred)


def dice_coef_SSVM_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + squared_hinge(y_true, y_pred)


def dice_coef_LGC_loss(y_true, y_pred):
    return dice_coef_loss(y_true, y_pred) + logcosh(y_true, y_pred)



