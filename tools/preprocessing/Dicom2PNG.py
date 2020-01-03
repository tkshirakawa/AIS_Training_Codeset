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
import math
import glob
import numpy as np
import pydicom
import cv2




if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a directory containing case directories')
    print('  argv[2] : Image file type, .jpg or .png')
    print('  argv[3] : Keyword to find directory for image (x data)')
    print('  argv[4] : Keyword to find directory for mask (y data)')
    print('  argv[5] : Path to a directory to save results in it')
    sys.exit()




def computeImage(pixel_array, wl, ww, scaled):

    if pixel_array.max() > 0:

        h, w = pixel_array.shape
        dd = int(max(h, w) / 20)
        if h > w:
            dtop = dd
            dbot = dd
            dlft = int((h - w) / 2) + dd
            drgt = (h - w) - dlft + 2 * dd
        elif h < w:
            dlft = dd
            drgt = dd
            dbot = int((w - h) / 2) + dd
            dtop = (w - h) - dbot + 2 * dd
        else:
            dtop = dd
            dbot = dd
            dlft = dd
            drgt = dd

        buf = pixel_array.astype(float)
        buf = cv2.copyMakeBorder(buf, dtop, dbot, dlft, drgt, cv2.BORDER_CONSTANT, value=[0.])
        if scaled:
            buf = cv2.resize(buf, dsize=(200, 200), interpolation=cv2.INTER_LANCZOS4)
            lo = wl - ww / 2.
            # hi = wl + ww / 2.
            buf = np.clip(255. * (buf - lo) / ww, 0., 255.)
        else:
            buf = cv2.resize(buf, dsize=(200, 200), interpolation=cv2.INTER_AREA)
            buf[buf <= 0.1] = 0.
            buf[buf > 0.1] = 255.
            # th = 0.8 * buf.min() + 0.2 * buf.max()
            # buf[buf <= th] = 0.
            # buf[buf > th] = 255.
        return np.uint8(buf)

    else:
        return np.zeros((200,200), dtype=np.uint8())




contentsList = glob.glob(os.path.join(sys.argv[1], '*'))

for directory in contentsList:

    if os.path.isdir(directory):

        print('---------------------------------------------------')
        print(directory)
    
        fileListX = glob.glob(os.path.join(directory, '*'+sys.argv[3]+'*', '*.dcm'))
        fileListY = glob.glob(os.path.join(directory, '*'+sys.argv[4]+'*', '*.dcm'))
        lx = len(fileListX)
        ly = len(fileListY)

        print('x: {0}, count: {1}'.format(sys.argv[3], lx))
        print('y: {0}, count: {1}'.format(sys.argv[4], ly))
    
        if lx != ly or lx == 0 or ly == 0:
            print('ERROR: File counts must be same or non-zero.')
            continue


        print('Loading DICOM images and their z positions...')
        ximg = {}
        yimg = {}
        for i, x_dcm, y_dcm in zip(range(lx), fileListX, fileListY):
            xdata = pydicom.read_file(x_dcm)
            ydata = pydicom.read_file(y_dcm)
            # xpos = str(xdata.ImagePositionPatient[2])
            # ypos = str(ydata.ImagePositionPatient[2])
            xpos = math.floor(100 * xdata.ImagePositionPatient[2]) + 100000
            ypos = math.floor(100 * ydata.ImagePositionPatient[2]) + 100000
            print('{0} - x {1}, y {2}'.format(i+1, xpos, ypos))
            # print(xdata[0x0028, 0x1050].value, xdata[0x0028, 0x1051].value)
            # print(ydata[0x0028, 0x1050].value, ydata[0x0028, 0x1051].value)
            # xlo, xhi = np.percentile(xdata.pixel_array, (2.5, 97.5))
            # ylo, yhi = np.percentile(ydata.pixel_array, (2.5, 97.5))
            # ximg[xpos] = computeImage(exposure.rescale_intensity(xdata.pixel_array, in_range=(xlo, xhi)) )
            # yimg[ypos] = computeImage(exposure.rescale_intensity(ydata.pixel_array, in_range=(ylo, yhi)) )
            ximg[xpos] = computeImage(xdata.pixel_array, xdata[0x0028, 0x1050].value, xdata[0x0028, 0x1051].value, True)
            yimg[ypos] = computeImage(ydata.pixel_array, 0, 1, False)


        print('Sorting DICOM images to their z positions...')
        ximg2 = sorted(ximg.items())
        yimg2 = sorted(yimg.items())


        print('Saving sorted DICOM images as {0} files in...'.format(sys.argv[2]))
        dirname, basename = os.path.split(directory)
        dirname = os.path.basename(dirname)
        save_dir = os.path.join(sys.argv[5], 'case_'+dirname+'_'+basename)
        print(save_dir)
        xdir = os.path.join(save_dir, 'x')
        ydir = os.path.join(save_dir, 'y')
        os.makedirs(xdir, exist_ok=True)
        os.makedirs(ydir, exist_ok=True)
        for i in range(lx):
            xpos = ximg2[i][0]
            ypos = yimg2[i][0]
            if xpos == ypos:
                print('{0} - position x == y : {1}'.format(i+1, xpos))
                cv2.imwrite(os.path.join(xdir, str(i+1).zfill(4)+sys.argv[2]), ximg2[i][1])
                cv2.imwrite(os.path.join(ydir, str(i+1).zfill(4)+sys.argv[2]), yimg2[i][1])
            else:
                print('{0} - position x != y : x {1}, y {2} : ERROR'.format(i+1, xpos, ypos))


