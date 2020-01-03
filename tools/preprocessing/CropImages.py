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


ncrop = 5
margin = 10        # For cropping
margnP = 100 + margin
margnM = 100 - margin
mrgnM2 = 200 - margin
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
    img_crop.append(cv2.copyMakeBorder(img_src[50:150,        50:150],        50,     50,     50,     50,     cv2.BORDER_CONSTANT, value=black))
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



