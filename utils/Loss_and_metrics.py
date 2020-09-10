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

# from tensorflow.python.framework import ops
# from tensorflow.python.ops import math_ops as mop

from keras import backend as K
from keras.losses import mean_squared_error, logcosh, kullback_leibler_divergence

##### For TensorFlow v2 #####
# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras.losses import mean_squared_error, logcosh, kullback_leibler_divergence




# Axis for image
# IMPORTANT to complete calculation in each image plane in several batches or samples 
# NOTE: K.image_data_format() must be 'channels_last'! = (samples, height, width, channels) *channels=1 for grayscale images in this model
# NOTE: In the following metrics and losses, the tensor 'y_XXXX' has the dimension of (batch size, height, width, channels=1)
img = (-3,-2,-1)


# Clip values between 0 and 1 with keeping its gradient
# 0.0 if x < 0.0; 1.0 if x > 1.0; x if 0.0 <= x <= 1.0
def hard_sigmoid_clip(val): return tf.keras.activations.hard_sigmoid(5.0 * (val - 0.5))




###
### Metrics
###

# IoU: 0.0 [bad] - 1.0 [good]
def iou_score(y_true, y_pred):
    t = hard_sigmoid_clip(y_true)       # Dimension = (batch size, height, width, channels=1)
    p = hard_sigmoid_clip(y_pred)
    u = hard_sigmoid_clip(t + p)
    truth = K.sum(t, axis=img)          # Dimension = (batch size)
    predc = K.sum(p, axis=img)
    union = K.sum(u, axis=img)
    count = K.max(t, axis=img)
    return K.sum((truth + predc - union) / (union + K.epsilon())) / K.sum(count)    # Scalar: The mean value alongside the batches where y_true is not zero


# Dice: 0.0 [bad] - 1.0 [good]
def dice_coef(y_true, y_pred):
    t = hard_sigmoid_clip(y_true)
    p = hard_sigmoid_clip(y_pred)
    u = hard_sigmoid_clip(t + p)
    truth = K.sum(t, axis=img)
    predc = K.sum(p, axis=img)
    union = K.sum(u, axis=img)
    count = K.max(t, axis=img)
    t_or_p = truth + predc
    t_and_p_2 = 2.0 * (t_or_p - union)
    return K.sum(t_and_p_2 / (t_or_p + K.epsilon())) / K.sum(count)


# RoU: residual over union: 0.0 [good] - 1.0 [bad]
# The residual is uncovered residual area in grand-truth. RoU = grand-truth - intersection
def rou(y_true, y_pred):
    t = hard_sigmoid_clip(y_true)
    p = hard_sigmoid_clip(y_pred)
    u = hard_sigmoid_clip(t + p)
    truth = K.sum(t, axis=img)
    predc = K.sum(p, axis=img)
    union = K.sum(u, axis=img)
    count = K.max(t, axis=img)
    return K.sum((union - predc) / (union + K.epsilon())) / K.sum(count)


# FPoET: false positives on empty truth: 0.0 [good] - 1.0 [bad]
def fpoet(y_true, y_pred):
    empty = 1.0 - K.max(hard_sigmoid_clip(y_true), axis=img)        # Dimension = (batch size), 1.0 if a slice of y_true is empty (= all the pixels are zero)
    predc = K.sum(hard_sigmoid_clip(y_pred), axis=img)              # Dimension = (batch size), the area of prediction pixels in each slice of y_true
    area  = K.sum(K.ones_like(K.max(y_true, axis=-1), dtype='float32'), axis=(-2,-1))   # Dimension = (batch size)
    return K.sum(empty * predc / area) / (K.sum(empty) + K.epsilon())




###
### Losses
###

def mean_squared_error_loss(y_true, y_pred):
    t = hard_sigmoid_clip(y_true)                       # Dimension = (batch size, height, width, channels=1)
    p = hard_sigmoid_clip(y_pred)
    return K.mean(mean_squared_error(t, p), axis=None)  # Scalar: The mean value alongside all the axes


def Huber_loss(y_true, y_pred):
    t = hard_sigmoid_clip(y_true)
    p = hard_sigmoid_clip(y_pred)
    return K.mean(tf.compat.v1.losses.huber_loss(t, p), axis=None)


def LogCosh_loss(y_true, y_pred):
    t = hard_sigmoid_clip(y_true)
    p = hard_sigmoid_clip(y_pred)
    return K.mean(logcosh(t, p), axis=None)


def kullback_leibler_divergence_loss(y_true, y_pred):
    t = hard_sigmoid_clip(y_true)
    p = hard_sigmoid_clip(y_pred)
    return K.mean(kullback_leibler_divergence(t, p), axis=None)


def MSE_loss_w_iou_score(y_true, y_pred):
    return 1.0 - iou_score(y_true, y_pred) + mean_squared_error_loss(y_true, y_pred)


def MSE_loss_w_iou_score_fpoet(y_true, y_pred):
    return 1.0 - iou_score(y_true, y_pred) + fpoet(y_true, y_pred) + mean_squared_error_loss(y_true, y_pred)


def MSE_loss_w_dice_coef(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred) + mean_squared_error_loss(y_true, y_pred)


def MSE_loss_w_dice_coef_fpoet(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred) + fpoet(y_true, y_pred) + mean_squared_error_loss(y_true, y_pred)


def Huber_loss_w_iou_score(y_true, y_pred):
    return 1.0 - iou_score(y_true, y_pred) + Huber_loss(y_true, y_pred)


def LogCosh_loss_w_iou_score(y_true, y_pred):
    return 1.0 - iou_score(y_true, y_pred) + LogCosh_loss(y_true, y_pred)


def LogCosh_loss_w_iou_score_fpoet(y_true, y_pred):
    return 1.0 - iou_score(y_true, y_pred) + fpoet(y_true, y_pred) + LogCosh_loss(y_true, y_pred)


def LogCosh_loss_w_dice_coef(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred) + LogCosh_loss(y_true, y_pred)


def LogCosh_loss_w_dice_coef_fpoet(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred) + fpoet(y_true, y_pred) + LogCosh_loss(y_true, y_pred)


def KLD_loss_w_iou_score(y_true, y_pred):
    return 1.0 - iou_score(y_true, y_pred) + kullback_leibler_divergence_loss(y_true, y_pred)


def KLD_loss_w_dice_coef(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred) + kullback_leibler_divergence_loss(y_true, y_pred)



