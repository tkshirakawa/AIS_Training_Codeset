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




if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a CSV file for training image paths')
    print('  argv[2] : Path to a CSV file for validation image paths')
    print('  argv[3] : Path to a directory to save h5 image dataset in it')
    print('  NOTE: Input none to pass argv[1] or argv[2]')
    sys.exit()



def ReadImage_and_CreateH5(csvfile, dataname, shuffle=True):

    if csvfile == 'none':
        return

    with open(csvfile, 'r', newline='', encoding='utf-8_sig') as f:
        csvlength = sum(1 for line in f) - 1     # !!! First row is a header, skip !!!

    print(' ')
    print('Data name : {0}'.format(dataname))
    print('Path      : {0}'.format(csvfile))
    print('Images    : {0}'.format(csvlength))

    # Open the CSV file
    with open(csvfile, 'r', newline='', encoding='utf-8_sig') as f:
        reader = csv.reader(f)
        next(reader)                                # !!! First row is a header, skip !!!
        paths = [row for row in reader]


    # Store image data as 'uint8' to save memory consumption
    XX = np.empty((csvlength, 200, 200, 1), dtype=np.uint8)
    YY = np.empty((csvlength, 200, 200, 1), dtype=np.uint8)
    index = np.arange(csvlength)        # [0,1,2, ...]
    if shuffle:
        np.random.shuffle(index)

    # To improve loading speed for huge set of images, loading ten sets per one loop
    lastnum_for10set = 10 * (csvlength // 10)
    nimg_foramark = max(1, csvlength // 100)
    for i in range(0, lastnum_for10set, 10):
        XX[index[i  ],:,:,0] = cv2.imread(paths[i  ][0], -1)
        XX[index[i+1],:,:,0] = cv2.imread(paths[i+1][0], -1)
        XX[index[i+2],:,:,0] = cv2.imread(paths[i+2][0], -1)
        XX[index[i+3],:,:,0] = cv2.imread(paths[i+3][0], -1)
        XX[index[i+4],:,:,0] = cv2.imread(paths[i+4][0], -1)
        XX[index[i+5],:,:,0] = cv2.imread(paths[i+5][0], -1)
        XX[index[i+6],:,:,0] = cv2.imread(paths[i+6][0], -1)
        XX[index[i+7],:,:,0] = cv2.imread(paths[i+7][0], -1)
        XX[index[i+8],:,:,0] = cv2.imread(paths[i+8][0], -1)
        XX[index[i+9],:,:,0] = cv2.imread(paths[i+9][0], -1)
        YY[index[i  ],:,:,0] = cv2.imread(paths[i  ][1], -1)
        YY[index[i+1],:,:,0] = cv2.imread(paths[i+1][1], -1)
        YY[index[i+2],:,:,0] = cv2.imread(paths[i+2][1], -1)
        YY[index[i+3],:,:,0] = cv2.imread(paths[i+3][1], -1)
        YY[index[i+4],:,:,0] = cv2.imread(paths[i+4][1], -1)
        YY[index[i+5],:,:,0] = cv2.imread(paths[i+5][1], -1)
        YY[index[i+6],:,:,0] = cv2.imread(paths[i+6][1], -1)
        YY[index[i+7],:,:,0] = cv2.imread(paths[i+7][1], -1)
        YY[index[i+8],:,:,0] = cv2.imread(paths[i+8][1], -1)
        YY[index[i+9],:,:,0] = cv2.imread(paths[i+9][1], -1)
        if (i % 100 == 0):
            sys.stdout.write('\rLoading   : %d ' % i + '#'*(i//nimg_foramark))
            sys.stdout.flush()

    for i in range(lastnum_for10set, csvlength, 1):
        XX[index[i],:,:,0] = cv2.imread(paths[i][0], -1)
        YY[index[i],:,:,0] = cv2.imread(paths[i][1], -1)
    print('\rLoaded    : %d ' % csvlength + '#'*100)


    # Create h5 file
    with h5py.File(os.path.join(sys.argv[3], dataname+'.h5'), 'w') as f:
        f.create_dataset(dataname+'_X', data=XX)
        f.create_dataset(dataname+'_Y', data=YY)
        f.flush()




# Read images and create h5 dataset files for training and validation
##########################################################################

ReadImage_and_CreateH5(sys.argv[1], 'image_training', shuffle=True)
ReadImage_and_CreateH5(sys.argv[2], 'image_validation', shuffle=True)

