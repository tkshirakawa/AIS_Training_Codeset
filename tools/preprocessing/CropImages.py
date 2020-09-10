'''
    Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com
    
    
    Released under the MIT license.
    https://opensource.org/licenses/mit-license.php

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


import sys
import os

import cv2
import numpy as np
import glob
import csv
import shutil




if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Image file type, .jpg or .png')
    print('  argv[2] : Path to a directory of images')
    print('  argv[3] : Images or mask images? Input image or mask')
    sys.exit()


    

if sys.argv[1] != '.jpg' and sys.argv[1] != '.png':
    print('### ERROR: Use jpg or png for input and result images!')
    sys.exit()


img_size = 512      # 200
img_size1 = 128      # 50
img_size2 = 256      # 100
img_size3 = 384      # 150

ncrop = 5
margin = 10         # For cropping
margnP = img_size2 + margin
margnM = img_size2 - margin
mrgnM2 = img_size - margin
black = [0]

files = glob.glob(os.path.join(sys.argv[2], '[0-9][0-9][0-9][0-9]'+sys.argv[1]))

dirpath = []
for i in range(ncrop):
    dirpath.append(sys.argv[2]+'_CP{0}'.format(i))
    if os.path.isdir(dirpath[i]):
        shutil.rmtree(dirpath[i])
    os.makedirs(dirpath[i])


csvFile = open(os.path.join(os.path.dirname(sys.argv[2]), 'Cropped images ('+sys.argv[3]+').csv'), 'w', newline='', encoding='utf-8')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(['File path to '+sys.argv[3]])
listData = []

for fpath in files:

    filename = os.path.basename(fpath)
    img_src = cv2.imread(fpath, -1)
    print('Input : ' + fpath)

    # Crop
    img_crop = []
    img_crop.append(cv2.copyMakeBorder(img_src[img_size1:img_size3, img_size1:img_size3], img_size1, img_size1, img_size1, img_size1, cv2.BORDER_CONSTANT, value=black))
    img_crop.append(cv2.copyMakeBorder(img_src[margin:margnP, margin:margnP], margin, margnM, margin, margnM, cv2.BORDER_CONSTANT, value=black))
    img_crop.append(cv2.copyMakeBorder(img_src[margin:margnP, margnM:mrgnM2], margin, margnM, margnM, margin, cv2.BORDER_CONSTANT, value=black))
    img_crop.append(cv2.copyMakeBorder(img_src[margnM:mrgnM2, margin:margnP], margnM, margin, margin, margnM, cv2.BORDER_CONSTANT, value=black))
    img_crop.append(cv2.copyMakeBorder(img_src[margnM:mrgnM2, margnM:mrgnM2], margnM, margin, margnM, margin, cv2.BORDER_CONSTANT, value=black))

    if sys.argv[3] == 'mask':
        for i in range(ncrop):
            cpfpath = os.path.join(dirpath[i], filename)
            listData.append([cpfpath])
            cv2.imwrite(cpfpath, cv2.threshold(img_crop[i], 127, 255, cv2.THRESH_BINARY)[1])
    else:
        for i in range(ncrop):
            cpfpath = os.path.join(dirpath[i], filename)
            listData.append([cpfpath])
            cv2.imwrite(cpfpath, img_crop[i])

csvWriter.writerows(listData)
csvFile.close()



