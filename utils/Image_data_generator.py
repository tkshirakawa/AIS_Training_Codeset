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


import os
import sys
import csv
import cv2
import numpy as np
import h5py

from keras import backend as K


# Image data generator for CSV lists
#####################################################################
# Read a list (first row for header, two column for data) from .csv file, and load images.
# You should not shuffle the rows in .csv file, or image loading speed will be decreased.
# Images must be 200x200 gray-scale without alpha values.

class ImageDataGenerator_CSV_with_Header:

    length_ = 0
    X_ = []
    Y_ = []

    def __init__(self, dataname, datafile, shuffle=True):
        with open(datafile, 'r', newline='', encoding='utf-8_sig') as f:
            self.length_ = sum(1 for line in f) - 1     # !!! First row is a header, skip !!!

        print('Data name : {0}'.format(dataname))
        print('Path      : {0}'.format(datafile))
        print('Counts    : {0}'.format(self.length_))

        # Open the CSV file
        with open(datafile, 'r', newline='', encoding='utf-8_sig') as f:
            reader = csv.reader(f)
            next(reader)                                # !!! First row is a header, skip !!!
            paths = [row for row in reader]

        # Store image data as 'uint8' to save memory consumption
        self.X_ = np.empty((self.length_, 200, 200, 1), dtype=np.uint8)
        self.Y_ = np.empty((self.length_, 200, 200, 1), dtype=np.uint8)
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


    def length(self):
        return self.length_


    def flow(self, rescale=1.0, batch_size=32):
        n_batch = self.length_ // batch_size
        while True:
            for i in range(n_batch):
                X_batch = K.cast_to_floatx(rescale * self.X_[i*batch_size:(i+1)*batch_size])
                Y_batch = K.cast_to_floatx(rescale * self.Y_[i*batch_size:(i+1)*batch_size])
                yield (X_batch, Y_batch)



# Image data generator for h5 file
#####################################################################

class ImageDataGenerator_h5_Dataset:

    length_ = 0
    X_ = []
    Y_ = []

    def __init__(self, dataname, datafile):
        f = h5py.File(datafile, 'r')

        self.X_ = f[dataname+'_X'].value
        self.Y_ = f[dataname+'_Y'].value
        self.length_ = self.X_.shape[0]
        
        print('Data name      : {0}'.format(dataname))
        print('Path           : {0}'.format(datafile))
        print('Counts         : {0}'.format(self.length_))
        print('Shape and type : {0}, {1}'.format(self.X_.shape, self.X_.dtype))

        f.close()


    def length(self):
        return self.length_


    def flow(self, rescale=1.0, batch_size=32):
        n_batch = self.length_ // batch_size
        while True:
            for i in range(n_batch):
                X_batch = K.cast_to_floatx(rescale * self.X_[i*batch_size:(i+1)*batch_size])
                Y_batch = K.cast_to_floatx(rescale * self.Y_[i*batch_size:(i+1)*batch_size])
                yield (X_batch, Y_batch)

