'''
    Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com
    
    
    Released under the MIT license.
    https://opensource.org/licenses/mit-license.php

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


import os
import sys
import csv
import cv2
import h5py
import math
import numpy as np


import tensorflow as tf
from keras import backend as K
# from keras.utils import Sequence

##### For TensorFlow v2 #####
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import backend as K
# from tensorflow.keras.utils import Sequence


'''
    For TensorFlow 2.0
    Image data generator should be a subclass of keras.utils.Sequence that generates batch data.

    See:
    https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence
'''


# Image data generator for CSV lists
#####################################################################
# Read a list (first row for header, two column for data) from .csv file, and load images.
# You should not shuffle the rows in .csv file, or image loading speed will be decreased.
# Images must be gray-scale without alpha values.

# class ImageDataGenerator_CSV_with_Header(Sequence):       ##### For TensorFlow v2 #####
class ImageDataGenerator_CSV_with_Header():

    def __init__(self, dataname, datafile, batch_size=32, rescale=1.0, shuffle=True):
        self.batch_size = batch_size
        self.rescale = rescale

        with open(datafile, 'r', newline='', encoding='utf-8_sig') as f:
            self.length_ = sum(1 for line in f) - 1     # !!! First row is a header, skip !!!

        # Open the CSV file
        with open(datafile, 'r', newline='', encoding='utf-8_sig') as f:
            reader = csv.reader(f)
            next(reader)                                # !!! First row is a header, skip !!!
            paths = [row for row in reader]

        # Image dimensions
        img_check = cv2.imread(paths[0][0], -1)
        height, width = img_check.shape[:2]

        print('Data name      : {0}'.format(dataname))
        print('Path           : {0}'.format(datafile))
        print('Counts         : {0}'.format(self.length_))
        print('H,W            : {0}x{1}'.format(height, width))
        print('Batch size     : {0}'.format(self.batch_size))
        print('Pixel scaling  : {0}'.format(self.rescale))

        # Store image data as 'uint8' to save memory consumption
        self.X_ = np.empty((self.length_, height, width, 1), dtype=np.uint8)
        self.Y_ = np.empty((self.length_, height, width, 1), dtype=np.uint8)
        index = np.arange(self.length_)        # [0,1,2, ...]
        if shuffle:
            np.random.shuffle(index)

        # To improve loading speed for huge set of images, loading ten sets per one loop
        lastnum_for10set = 10 * (self.length_ // 10)
        nimg_foramark = max(1, self.length_ // 100)
        for i in range(0, lastnum_for10set, 10):
            self.X_[index[i  ],:,:,0] = cv2.imread(paths[i  ][0], -1)
            self.X_[index[i+1],:,:,0] = cv2.imread(paths[i+1][0], -1)
            self.X_[index[i+2],:,:,0] = cv2.imread(paths[i+2][0], -1)
            self.X_[index[i+3],:,:,0] = cv2.imread(paths[i+3][0], -1)
            self.X_[index[i+4],:,:,0] = cv2.imread(paths[i+4][0], -1)
            self.X_[index[i+5],:,:,0] = cv2.imread(paths[i+5][0], -1)
            self.X_[index[i+6],:,:,0] = cv2.imread(paths[i+6][0], -1)
            self.X_[index[i+7],:,:,0] = cv2.imread(paths[i+7][0], -1)
            self.X_[index[i+8],:,:,0] = cv2.imread(paths[i+8][0], -1)
            self.X_[index[i+9],:,:,0] = cv2.imread(paths[i+9][0], -1)
            self.Y_[index[i  ],:,:,0] = cv2.imread(paths[i  ][1], -1)
            self.Y_[index[i+1],:,:,0] = cv2.imread(paths[i+1][1], -1)
            self.Y_[index[i+2],:,:,0] = cv2.imread(paths[i+2][1], -1)
            self.Y_[index[i+3],:,:,0] = cv2.imread(paths[i+3][1], -1)
            self.Y_[index[i+4],:,:,0] = cv2.imread(paths[i+4][1], -1)
            self.Y_[index[i+5],:,:,0] = cv2.imread(paths[i+5][1], -1)
            self.Y_[index[i+6],:,:,0] = cv2.imread(paths[i+6][1], -1)
            self.Y_[index[i+7],:,:,0] = cv2.imread(paths[i+7][1], -1)
            self.Y_[index[i+8],:,:,0] = cv2.imread(paths[i+8][1], -1)
            self.Y_[index[i+9],:,:,0] = cv2.imread(paths[i+9][1], -1)
            if i % 100 == 0:
                sys.stdout.write('\r  Loading : %d ' % i + '#'*(i//nimg_foramark))
                sys.stdout.flush()

        for i in range(lastnum_for10set, self.length_, 1):
            self.X_[index[i],:,:,0] = cv2.imread(paths[i][0], -1)
            self.Y_[index[i],:,:,0] = cv2.imread(paths[i][1], -1)
        print('\r  Loaded  : %d ' % self.length_ + '#'*100)


    # Number of batch in the Sequence       ##### For TensorFlow v2 #####
    # def __len__(self): return math.ceil(self.length_ / self.batch_size)


    # Gets batch at position index          ##### For TensorFlow v2 #####
    # def __getitem__(self, index):
    #     X_batch = K.cast_to_floatx(self.rescale * self.X_[index*self.batch_size : (index+1)*self.batch_size])
    #     Y_batch = K.cast_to_floatx(self.rescale * self.Y_[index*self.batch_size : (index+1)*self.batch_size])
    #     return (X_batch, Y_batch)


    def datalength(self): return self.length_


    def getdata(self):
        X_batch = K.cast_to_floatx(self.rescale * self.X_[0:])
        Y_batch = K.cast_to_floatx(self.rescale * self.Y_[0:])
        return (X_batch, Y_batch)


    def flow(self):
        n_batch = self.length_ // self.batch_size
        while True:
            for i in range(n_batch):
                X_batch = K.cast_to_floatx(self.rescale * self.X_[i*self.batch_size:(i+1)*self.batch_size])
                Y_batch = K.cast_to_floatx(self.rescale * self.Y_[i*self.batch_size:(i+1)*self.batch_size])
                yield (X_batch, Y_batch)




# Image data generator for h5 file
#####################################################################

# class ImageDataGenerator_h5_Dataset(Sequence):        ##### For TensorFlow v2 #####
class ImageDataGenerator_h5_Dataset():

    def __init__(self, dataname, datafile, batch_size=32, rescale=1.0):
        self.batch_size = batch_size
        self.rescale = rescale

        f = h5py.File(datafile, 'r')

        self.X_ = f[dataname+'_X'].value    # type = <class 'numpy.ndarray'>
        self.Y_ = f[dataname+'_Y'].value    # type = <class 'numpy.ndarray'>
        self.length_ = self.X_.shape[0]

        print('Data name      : {0}'.format(dataname))
        print('Path           : {0}'.format(datafile))
        print('Counts         : {0}'.format(self.length_))
        print('Shape/type (X) : {0} / {1}'.format(self.X_.shape, self.X_.dtype))
        print('Shape/type (Y) : {0} / {1}'.format(self.Y_.shape, self.Y_.dtype))
        print('Batch size     : {0}'.format(self.batch_size))
        print('Pixel scaling  : {0}'.format(self.rescale))

        f.close()


    # Number of batch in the Sequence       ##### For TensorFlow v2 #####
    # def __len__(self): return math.ceil(self.length_ / self.batch_size)


    # Gets batch at position index          ##### For TensorFlow v2 #####
    # def __getitem__(self, index):
    #     X_batch = K.cast_to_floatx(self.rescale * self.X_[index*self.batch_size : (index+1)*self.batch_size])
    #     Y_batch = K.cast_to_floatx(self.rescale * self.Y_[index*self.batch_size : (index+1)*self.batch_size])
    #     return (X_batch, Y_batch)


    def datalength(self): return self.length_


    def getdata(self):
        X_batch = K.cast_to_floatx(self.rescale * self.X_[0:])
        Y_batch = K.cast_to_floatx(self.rescale * self.Y_[0:])
        return (X_batch, Y_batch)


    def flow(self):
        n_batch = self.length_ // self.batch_size
        while True:
            for i in range(n_batch):
                X_batch = K.cast_to_floatx(self.rescale * self.X_[i*self.batch_size:(i+1)*self.batch_size])
                Y_batch = K.cast_to_floatx(self.rescale * self.Y_[i*self.batch_size:(i+1)*self.batch_size])
                yield (X_batch, Y_batch)



