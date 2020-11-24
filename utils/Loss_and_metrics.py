'''
    Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com
    
    
    Released under the MIT license.
    https://opensource.org/licenses/mit-license.php

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


# import numpy as np
import tensorflow as tf

from keras import backend as K
from keras.losses import mean_squared_error, logcosh, kullback_leibler_divergence

##### For TensorFlow v2 #####
#from tensorflow import keras
#from tensorflow.keras import backend as K
#from tensorflow.keras.losses import mean_squared_error, logcosh, kullback_leibler_divergence




# Axis for image
# IMPORTANT to complete calculation in each image plane in several batches or samples 
# NOTE: K.image_data_format() must be 'channels_last'! = (samples, height, width, channels) *channels=1 for grayscale images in this model
# NOTE: In the following metrics and losses, the tensor 'y_XXXX' has the dimension of (batch size, height, width, channels=1)
HWC = (-3,-2,-1)
HW  = (-2,-1)


# Clip values between 0 and 1 with keeping its gradient
# 0.0 if x < 0.0; 1.0 if x > 1.0; x if 0.0 <= x <= 1.0
def hard_sigmoid_clip(val): return tf.keras.activations.hard_sigmoid(5.0 * (val - 0.5))




'''
Metrics
'''

# IoU: 0.0 [bad] - 1.0 [good]
def iou_score_base(y_true, y_pred):
    t = hard_sigmoid_clip(y_true)       # Dimension = (batch size, height, width, channels=1)
    p = hard_sigmoid_clip(y_pred)
    u = hard_sigmoid_clip(t + p)
    truth = K.sum(t, axis=HWC)          # Dimension = (batch size)
    predc = K.sum(p, axis=HWC)
    union = K.sum(u, axis=HWC)
    return (truth + predc - union) / (union + K.epsilon()), K.sum(K.max(t, axis=HWC))
    # IoU score of each batch (the size = [batch]), the count of non-zero groundtruth in the batch (my be zero, the size = [1])

def iou_score(y_true, y_pred):
    iou, count = iou_score_base(y_true, y_pred)     # 'count' may be zero!
    return K.sum(iou) / K.maximum(count, 1.0)       # Scalar: The mean value alongside the batches where y_true is not zero


# Dice: 0.0 [bad] - 1.0 [good]
def dice_coef_base(y_true, y_pred):
    t = hard_sigmoid_clip(y_true)       # Dimension = (batch size, height, width, channels=1)
    p = hard_sigmoid_clip(y_pred)
    u = hard_sigmoid_clip(t + p)
    truth = K.sum(t, axis=HWC)
    predc = K.sum(p, axis=HWC)
    union = K.sum(u, axis=HWC)
    t_or_p = truth + predc
    t_and_p_2 = 2.0 * (t_or_p - union)
    return t_and_p_2 / (t_or_p + K.epsilon()), K.sum(K.max(t, axis=HWC))
    # Dice coef. of each batch (the size = [batch]), the count of non-zero groundtruth in the batch (my be zero, the size = [1])

def dice_coef(y_true, y_pred):
    dice, count = dice_coef_base(y_true, y_pred)    # 'count' may be zero!
    return K.sum(dice) / K.maximum(count, 1.0)      # Scalar: The mean value alongside the batches where y_true is not zero


# RoU: residual over union: 0.0 [good] - 1.0 [bad]
# The residual is uncovered residual area in grand-truth. RoU = grand-truth - intersection
def rou_base(y_true, y_pred):
    t = hard_sigmoid_clip(y_true)       # Dimension = (batch size, height, width, channels=1)
    p = hard_sigmoid_clip(y_pred)
    u = hard_sigmoid_clip(t + p)
    truth = K.sum(t, axis=HWC)
    predc = K.sum(p, axis=HWC)
    union = K.sum(u, axis=HWC)
    return (union - predc) / (union + K.epsilon()), K.sum(K.max(t, axis=HWC))
    # return K.sum((union - predc) / (union + K.epsilon())) / K.sum(K.max(t, axis=HWC))

def rou(y_true, y_pred):
    rou, count = rou_base(y_true, y_pred)           # 'count' may be zero!
    return K.sum(rou) / K.maximum(count, 1.0)       # Scalar: The mean value alongside the batches where y_true is not zero


# FPoET: false positives on empty truth: 0.0 [good] - 1.0 [bad]
def fpoet(y_true, y_pred):
    empty = 1.0 - K.max(hard_sigmoid_clip(y_true), axis=HWC)                        # Dimension = (batch size), 1.0 if a slice of y_true is empty (= all the pixels are zero)
    predc = K.sum(hard_sigmoid_clip(y_pred), axis=HWC)                              # Dimension = (batch size), the area of prediction pixels in each slice of y_true
    area  = K.sum(K.ones_like(K.max(y_true, axis=-1), dtype=K.floatx()), axis=HW)   # Dimension = (batch size), the pixel area of each image
    return K.sum(empty * predc / area) / (K.sum(empty) + K.epsilon())




'''
Losses
'''

# Helper method
# NOTE: Combining IoU/Dice into a loss function will contribute to rapid increase of IoU/Dice scores,
# BUT it may also make those scores unstable during epochs
def combination_loss(funcA, mean_axis, funcB_base, auto_ratio, y_true, y_pred):
    if mean_axis == 'none': a = funcA(y_true, y_pred)                           # Dimension = (batch size)
    else:                   a = K.mean(funcA(y_true, y_pred), axis=mean_axis)   # Dimension = (batch size)
    b, c = funcB_base(y_true, y_pred)                                           # Dimension = (batch size), scalar(may be zero!)
    # b = 1.0 - K.square(b)
    # return (K.sum(a) + K.sum(b)) / K.maximum(c, 1.0)
    return K.sum(a + 1.0 - b) / K.maximum(c, 1.0)


def MSE_loss(y_true, y_pred):               return mean_squared_error(y_true, y_pred)
def MSE_loss_w_iou_score(y_true, y_pred):   return combination_loss(mean_squared_error, HW, iou_score_base, True, y_true, y_pred)
def MSE_loss_w_dice_coef(y_true, y_pred):   return combination_loss(mean_squared_error, HW, dice_coef_base, True, y_true, y_pred)

def LogCosh_loss(y_true, y_pred):               return logcosh(y_true, y_pred)
def LogCosh_loss_w_iou_score(y_true, y_pred):   return combination_loss(logcosh, HW, iou_score_base, True, y_true, y_pred)
def LogCosh_loss_w_dice_coef(y_true, y_pred):   return combination_loss(logcosh, HW, dice_coef_base, True, y_true, y_pred)

def KLD_loss(y_true, y_pred):               return kullback_leibler_divergence(y_true, y_pred)
def KLD_loss_w_iou_score(y_true, y_pred):   return combination_loss(kullback_leibler_divergence, HW, iou_score_base, True, y_true, y_pred)
def KLD_loss_w_dice_coef(y_true, y_pred):   return combination_loss(kullback_leibler_divergence, HW, dice_coef_base, True, y_true, y_pred)


'''
Focal Loss

Original work:
    Tsung-Yi Lin, Priya Goyal, Ross Girshick, Kaiming He, Piotr Dollár,
    Focal Loss for Dense Object Detection,
    arXiv:1708.02002v2
    Submitted on 7 Aug 2017 (v1), last revised 7 Feb 2018 (this version, v2)
    https://arxiv.org/abs/1708.02002v2
'''
def focal_CE_loss_base(y_true, y_pred, use_count, alpha=0.25, gamma=2.0):
    t = hard_sigmoid_clip(y_true)
    p = K.sigmoid(y_pred)
    t_inv = 1.0 - t
    p_t = t * p     + t_inv * (1.0 - p)
    a_t = t * alpha + t_inv * (1.0 - alpha)
    fl  = -a_t * K.pow(1.0 - p_t, gamma) * K.log(p_t)
    if use_count: return K.mean(fl, axis=HWC), K.sum(K.max(t, axis=HWC))
    else:         return K.mean(fl, axis=HWC)


def focal_CE_loss(y_true, y_pred):
    fl, count = focal_CE_loss_base(y_true, y_pred, use_count=True)      # 'count' may be zero!
    return K.sum(fl) / K.maximum(count, 1.0)                            # Scalar: The mean value alongside the batches where y_true is not zero


def focal_CE_loss_for_batch(y_true, y_pred):    return focal_CE_loss_base(y_true, y_pred, use_count=False)
def focal_CE_loss_w_iou_score(y_true, y_pred):  return combination_loss(focal_CE_loss_for_batch, 'none', iou_score_base, True, y_true, y_pred)
def focal_CE_loss_w_dice_coef(y_true, y_pred):  return combination_loss(focal_CE_loss_for_batch, 'none', dice_coef_base, True, y_true, y_pred)


'''
Constrained Focal Loss

Original work:
    Y. Zhao, F. Lin, S. Liu, Z. Hu, H. Li and Y. Bai.
    Constrained-Focal-Loss Based Deep Learning for Segmentation of Spores.
    in IEEE Access, vol. 7, pp. 165029-165038, 2019, doi: 10.1109/ACCESS.2019.2953085.
    https://ieeexplore.ieee.org/document/8896836
'''
def constrained_focal_CE_loss_base(y_true, y_pred, use_count, beta=2.5):
    t = hard_sigmoid_clip(y_true)                                                   # Dimension = (batch size, height, width, channels=1)
    p = K.sigmoid(y_pred)                                                           # Dimension = (batch size, height, width, channels=1)
    no   = K.maximum(K.sum(t, axis=HWC, keepdims=True), 1.0)                        # Dimension = (batch size, 1, 1, 1), nb = area - no
    area = K.sum(K.ones_like(t, dtype=K.floatx()), axis=HWC, keepdims=True)         # Dimension = (batch size, 1, 1, 1), the pixel area of each image
    p_t = t * p + (1.0 - t) * (1.0 - p)
    a_t = t * (area - no) / no + (1.0 - t)                                          # Broadcasted (batch size, 1, 1, 1) -> (batch size, height, width, channels=1)
    cfl = -K.pow(a_t, 1.0 / beta) * K.log(p_t)
    if use_count: return K.mean(cfl, axis=HWC), K.sum(K.max(t, axis=HWC))
    else:         return K.mean(cfl, axis=HWC)


def constrained_focal_CE_loss(y_true, y_pred):
    cfl, count = constrained_focal_CE_loss_base(y_true, y_pred, use_count=True)     # 'count' may be zero!
    return K.sum(cfl) / K.maximum(count, 1.0)                                       # Scalar: The mean value alongside the batches where y_true is not zero


def constrained_focal_CE_loss_for_batch(y_true, y_pred):    return constrained_focal_CE_loss_base(y_true, y_pred, use_count=False)
def constrained_focal_CE_loss_w_iou_score(y_true, y_pred):  return combination_loss(constrained_focal_CE_loss_for_batch, 'none', iou_score_base, True, y_true, y_pred)
def constrained_focal_CE_loss_w_dice_coef(y_true, y_pred):  return combination_loss(constrained_focal_CE_loss_for_batch, 'none', dice_coef_base, True, y_true, y_pred)


'''
Hausdorff Distance

Original work:
    Karimi D, Salcudean SE.
    Reducing the Hausdorff Distance in Medical Image Segmentation With Convolutional Neural Networks.
    IEEE Trans Med Imaging. 2020 Feb;39(2) 499-513. doi:10.1109/tmi.2019.2930068. PMID: 31329113.
'''
def hausdorff_distance_loss_base(y_true, y_pred, use_count, distance_limit=100, alpha=2.0):     # "... alpha between 1.0 and 3.0 leads to good results. ..."
    t = dt = dt_dilated = hard_sigmoid_clip(y_true)
    p = dp = dp_dilated = hard_sigmoid_clip(y_pred)
    # f = K.zeros(shape=(3, 3, 1))
    for i in range(distance_limit):
        # dilation2d is 10% faster than max_pool2d, but IoU and Dice are the better with max_pool2d
        # dt_dilated = tf.nn.dilation2d(dt_dilated, filter=f, strides=(1,1,1,1), rates=(1,1,1,1), padding='SAME')
        # dp_dilated = tf.nn.dilation2d(dp_dilated, filter=f, strides=(1,1,1,1), rates=(1,1,1,1), padding='SAME')
        dt_dilated = tf.nn.max_pool2d(dt_dilated, ksize=3, strides=1, padding='SAME', data_format='NHWC')
        dp_dilated = tf.nn.max_pool2d(dp_dilated, ksize=3, strides=1, padding='SAME', data_format='NHWC')
        dt += dt_dilated
        dp += dp_dilated
    dl = distance_limit + 1.0
    dt = (dl - dt) / dl
    dp = (dl - dp) / dl
    hddl  = K.square(t - p) * (K.pow(dt, alpha) + K.pow(dp, alpha))
    if use_count: return K.mean(hddl, axis=HWC), K.sum(K.max(t, axis=HWC))
    else:         return K.mean(hddl, axis=HWC)
    # The mean Hausdorff dist. of each batch (the size = [batch]), the count of non-zero groundtruth in the batch (my be zero, the size = [1])


def hausdorff_distance_loss(y_true, y_pred):
    hddl, count = hausdorff_distance_loss_base(y_true, y_pred, use_count=True)      # 'count' may be zero!
    return K.sum(hddl) / K.maximum(count, 1.0)                                      # Scalar: The mean value alongside the batches where y_true is not zero


def hausdorff_distance_loss_for_batch(y_true, y_pred):   return hausdorff_distance_loss_base(y_true, y_pred, use_count=False)
def hausdorff_distance_loss_w_iou_score(y_true, y_pred): return combination_loss(hausdorff_distance_loss_for_batch, 'none', iou_score_base, True, y_true, y_pred)
def hausdorff_distance_loss_w_dice_coef(y_true, y_pred): return combination_loss(hausdorff_distance_loss_for_batch, 'none', dice_coef_base, True, y_true, y_pred)




'''
Series combination loss
'''
selection_counter = 0

def series_combination_loss_for_batch(y_true, y_pred, interval=10):
    global selection_counter
    selector = (selection_counter // interval) % 6
    if selector   == 0: loss_calculator = K.mean(MSE_loss(y_true, y_pred), axis=HW)                 # Dimension = (batch size)
    elif selector == 1: loss_calculator = K.mean(LogCosh_loss(y_true, y_pred), axis=HW)             # Dimension = (batch size)
    elif selector == 2: loss_calculator = K.mean(KLD_loss(y_true, y_pred), axis=HW)                 # Dimension = (batch size)
    elif selector == 3: loss_calculator = focal_CE_loss_for_batch(y_true, y_pred)
    elif selector == 4: loss_calculator = constrained_focal_CE_loss_for_batch(y_true, y_pred)
    elif selector == 5: loss_calculator = hausdorff_distance_loss_for_batch(y_true, y_pred)
    selection_counter += 1
    return loss_calculator


def series_combination_loss_w_iou_score(y_true, y_pred): return combination_loss(series_combination_loss_for_batch, 'none', iou_score_base, True, y_true, y_pred)
def series_combination_loss_w_dice_coef(y_true, y_pred): return combination_loss(series_combination_loss_for_batch, 'none', dice_coef_base, True, y_true, y_pred)



