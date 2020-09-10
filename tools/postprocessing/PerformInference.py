'''
    Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com
    
    
    Released under the MIT license.
    https://opensource.org/licenses/mit-license.php

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


print('Description : Perform inference from images X (raw image).')

import sys
if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a trained Keras model.')
    print('  argv[2] : Path to a folder containing X images for input.')
    print('  argv[3] : Image file type to use, .jpg or .png.')
    print('  argv[4] : Path to a directory to save inference results (predicted images) in it.')
    print('  argv[5] : Path to a neural network file of Keras model (eg CV_net_Synapse.py) used for the training.')
    print('  argv[6] : Path to Train.py used for the training.')
    print('  Input images must be sqaure and gray-scale without alpha values.')
    sys.exit(0)


import os
import platform
import cv2
import numpy as np
import glob
from keras.models import load_model
import importlib.machinery as imm


if platform.system() == 'Darwin':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'


# Read X images
paths_imgX = glob.glob(os.path.join(sys.argv[2], '[0-9][0-9][0-9][0-9]'+sys.argv[3]))
imgCount = len(paths_imgX)

if imgCount <= 0:
    print('> Directories is EMPTY!!!')
    sys.exit(0)

# Size
tempImg = cv2.imread(paths_imgX[0], cv2.IMREAD_UNCHANGED)       # [height, width, channel] for color images, [height, width] for grayscale images
if tempImg.ndim != 2:
    print('> Grayscale images required!!!')
    sys.exit(0)

height, width = tempImg.shape
if height != width:
    print('> Square images required!!!')
    sys.exit(0)


print("Image size  : W={0}, H={1}".format(width, height))
print("Image count : {0}".format(imgCount))


# OpenCV(grayscale) = HEIGHT x WIDTH, Keras = HEIGHT x WIDTH x CHANNEL
imgX = np.zeros((imgCount, h, w, 1), dtype=np.float32)
for i, pathX in enumerate(paths_imgX):
    imgX[i,:,:,0] = cv2.imread(pathX, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.0


sys.path.append(os.path.dirname(sys.argv[6]))

NN = imm.SourceFileLoader('Keras_model_py', sys.argv[5]).load_module()
try:    custom_layers = NN.Custom_Layers()
except: custom_layers = {}      # Empty

Train_py = imm.SourceFileLoader('Train_py', sys.argv[6]).load_module()
custom_loss = Train_py.get_loss()
custom_metrics = Train_py.get_metrics()


# Predictionn
model = load_model(sys.argv[1], custom_objects=dict(**custom_loss, **custom_metrics, **custom_layers), compile=False)
model.summary()
imgY = model.predict(imgX, batch_size=16, verbose=1, steps=None)


print('Saving predicted images in... : ')
print(sys.argv[4])
for i, pathX in enumerate(paths_imgX):
    imgName = os.path.basename(pathX)
    cv2.imwrite(os.path.join(sys.argv[4], imgName), cv2.threshold((imgY[i,:,:,0] * 255.0).astype(np.uint8), 127, 255, cv2.THRESH_BINARY)[1])
    print("{0} : {1}".format(i+1, imgName))



