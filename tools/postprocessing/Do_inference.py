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


import sys
if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a parameter file of Keras')
    print('  argv[2] : Path to a folder containing X images')
    print('  argv[3] : Image file type, .jpg or .png')
    print('  argv[4] : Path to a directory to save predicted images in it')
    print('  argv[5] : Threshold ratio to make predicted images black-and-white')
    print('  argv[6] : Path to a Keras model code (.py) used for the training.')
    print('  argv[7] : Path to Train.py used for the training.')
    print('  Input images must be 200x200 gray-scale without alpha values')
    sys.exit()


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
pathX = glob.glob(os.path.join(sys.argv[2], '[0-9][0-9][0-9][0-9]'+sys.argv[3]))
nImgX = len(pathX)

if nImgX <= 0:
    print('> Directories is EMPTY!!!')
    sys.exit(0)


# OpenCV(grayscale) = HEIGHT x WIDTH, Keras = HEIGHT x WIDTH x CHANNEL
imgX = np.zeros((nImgX, 200, 200, 1), dtype=np.float32)
imgNames = []
for i in range(nImgX):
    imgX[i,:,:,0] = cv2.imread(pathX[i], 0).astype(np.float32) / 255.0
    imgNames.append(os.path.basename(pathX[i]))


sys.path.append(os.path.dirname(sys.argv[7]))

NN = imm.SourceFileLoader('Keras_model_py', sys.argv[6]).load_module()
try:    custom_layers = NN.Custom_Layers()
except: custom_layers = {}      # Empty

Train_py = imm.SourceFileLoader('Train_py', sys.argv[7]).load_module()
custom_loss = Train_py.get_loss()
custom_metrics = Train_py.get_metrics()


# Predictionn
model = load_model(sys.argv[1], custom_objects=dict(**custom_loss, **custom_metrics, **custom_layers), compile=False)
model.summary()
imgY = model.predict(imgX, batch_size=16, verbose=1, steps=None)   # X = HEIGHT x WIDTH x CHANNEL


print('Saving predicted images in... : ')
print(sys.argv[4])
# meanIoU = 0
for i in range(nImgX):
    print("{0} - {1} => {2}".format(i+1, pathX[i], imgNames[i]))
    result = (imgY[i,:,:,0] * 255.0).astype(np.uint8)
    cv2.imwrite(os.path.join(sys.argv[4], imgNames[i]), cv2.threshold(result, 255*float(sys.argv[5]), 255, cv2.THRESH_BINARY)[1])
#     imgY_AI_01 = imgY_AI > 127
#     imgY_01 = Y[i,:,:] > 127
#     iou_score = np.sum(np.logical_and(imgY_AI_01, imgY_01)) / np.sum(np.logical_or(imgY_AI_01, imgY_01))
#     meanIoU += iou_score
#     print('> {0} ; IoU = {1}'.format(imgNames[i], iou_score))

# print('> Mean IoU = {0}'.format(meanIoU/nPredImage))
# print('Done.')



