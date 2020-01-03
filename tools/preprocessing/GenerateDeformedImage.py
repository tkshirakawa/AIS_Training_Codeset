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
# import math

import cv2
import numpy as np
import glob
import shutil




if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Image file type; Input .jpg or .png')
    print('  argv[2] : Path to a directory of input images')
    print('  argv[3] : Is not used')
    print('  argv[4] : Images or mask images? Input image or mask')
    print('  argv[5] : Trimming width; positive value = top&bottom / negative value = right&left')
    sys.exit()




def makeDirsByDeletingOld(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)

# v1 = 6.5
# v2 = 1.1
# v3 = 127.5
# v4 = 127.5
# v1 = 6.5
# v2 = 1.2
# v3 = 127.5
# v4 = 127.5
# lut = [ np.uint8( np.clip( (255.0 / (1.0 + np.exp(-v1 * (i - v3) / 255.0)) - v4) * v2 + 127.5, 0, 255) ) for i in range(256)]
# lut = [ np.uint8( np.clip(255.0 / (1.0 + np.exp(-v1 * (i - 127.5) / 255.0)) + v2, 0, 255) ) for i in range(256)]

def shiftHisto(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(image)
    # return cv2.equalizeHist(image)
    # img_tmp = np.array([ lut[value] for value in image.flat], dtype=np.uint8)
    # return img_tmp.reshape(image.shape)




###################################################################3
def generateDeformedImage(format_ext, data_dir, dummy, modality, trimming):

    HW_padding = 10         # For height and width deformation
    trz_padding = 20        # For trapezoid deformation
    plg_padding = 15        # For parallelogram deformation
    rot_angle = 8           # For rotation
    black = [0]


    if format_ext != '.jpg' and format_ext != '.png':
        print('### ERROR: Use jpg or png for input and result images!')
        sys.exit()

    if modality == 'image':
        OC_dirpath = [data_dir+'_O', data_dir+'_C']   # [ Original, Contrast ]
    elif modality == 'mask':
        OC_dirpath = [data_dir+'_O']                     # [ Original ]
    else:
        print('### ERROR: Use image or mask!')
        sys.exit()


    files = glob.glob(os.path.join(data_dir, '[0-9][0-9][0-9][0-9]'+format_ext))
    if len(files) <= 0:
        print('### ERROR: No input files found!')
        sys.exit()


    clipTB = int(trimming)
    clipLR = 0
    if clipTB < 0:
        clipLR = -clipTB
        clipTB = 0


    for dirpath in OC_dirpath:

        makeDirsByDeletingOld(dirpath)
        makeDirsByDeletingOld(dirpath+'_HT')
        makeDirsByDeletingOld(dirpath+'_WD')
        makeDirsByDeletingOld(dirpath+'_DKP')
        makeDirsByDeletingOld(dirpath+'_DKM')
        makeDirsByDeletingOld(dirpath+'_HSR')
        makeDirsByDeletingOld(dirpath+'_HSL')
        makeDirsByDeletingOld(dirpath+'_RTP')
        makeDirsByDeletingOld(dirpath+'_RTM')


    for fpath in files:

        filename = os.path.basename(fpath)
        img_src_original = cv2.imread(fpath, -1)
        img_src_tmp = img_src_original[clipTB:200-clipTB, clipLR:200-clipLR]
        print('Input : ' + fpath)

        for dirpath in OC_dirpath:

            # Create color-changed images
            if dirpath[-1] == 'O':      # Original
                img_src = cv2.copyMakeBorder(img_src_tmp, clipTB, clipTB, clipLR, clipLR, cv2.BORDER_CONSTANT, value=black)
            elif dirpath[-1] == 'C':    # Contrast
                img_src = cv2.copyMakeBorder(shiftHisto(img_src_tmp), clipTB, clipTB, clipLR, clipLR, cv2.BORDER_CONSTANT, value=black)
            cv2.imwrite(os.path.join(dirpath, filename), img_src)

            # Height / width
            img_ht = cv2.resize(img_src, dsize=(200,200+2*HW_padding), interpolation=cv2.INTER_CUBIC)
            img_wd = cv2.resize(img_src, dsize=(200+2*HW_padding,200), interpolation=cv2.INTER_CUBIC)
            if modality == 'mask':
                cv2.imwrite(os.path.join(dirpath+'_HT', filename), cv2.threshold(img_ht[HW_padding:200+HW_padding,0:200], 127, 255, cv2.THRESH_BINARY)[1])
                cv2.imwrite(os.path.join(dirpath+'_WD', filename), cv2.threshold(img_wd[0:200,HW_padding:200+HW_padding], 127, 255, cv2.THRESH_BINARY)[1])
            else:
                cv2.imwrite(os.path.join(dirpath+'_HT', filename), img_ht[HW_padding:200+HW_padding,0:200])
                cv2.imwrite(os.path.join(dirpath+'_WD', filename), img_wd[0:200,HW_padding:200+HW_padding])

            # Trapezoid
            perspective1 = np.float32([[0, 200], [200, 200], [200, 0], [0, 0]])
            perspective2_P = np.float32([[-trz_padding, 200], [200+trz_padding, 200], [200, 0], [0, 0]])
            perspective2_M = np.float32([[0, 200], [200, 200], [200+trz_padding, 0], [-trz_padding, 0]])
            psp_matrix_P = cv2.getPerspectiveTransform(perspective1, perspective2_P)
            psp_matrix_M = cv2.getPerspectiveTransform(perspective1, perspective2_M)
            img_psp_P = cv2.warpPerspective(img_src, psp_matrix_P, (200,200))
            img_psp_M = cv2.warpPerspective(img_src, psp_matrix_M, (200,200))
            if modality == 'mask':
                cv2.imwrite(os.path.join(dirpath+'_DKP', filename), cv2.threshold(img_psp_P, 127, 255, cv2.THRESH_BINARY)[1])
                cv2.imwrite(os.path.join(dirpath+'_DKM', filename), cv2.threshold(img_psp_M, 127, 255, cv2.THRESH_BINARY)[1])
            else:
                cv2.imwrite(os.path.join(dirpath+'_DKP', filename), img_psp_P)
                cv2.imwrite(os.path.join(dirpath+'_DKM', filename), img_psp_M)

            # Parallelogram
            perspective1 = np.float32([[0, 200], [200, 200], [200, 0], [0, 0]])
            perspective2_P = np.float32([[-plg_padding, 200], [200, 200], [200+plg_padding, 0], [0, 0]])
            perspective2_M = np.float32([[0, 200], [200+plg_padding, 200], [200, 0], [-plg_padding, 0]])
            psp_matrix_P = cv2.getPerspectiveTransform(perspective1, perspective2_P)
            psp_matrix_M = cv2.getPerspectiveTransform(perspective1, perspective2_M)
            img_psp_P = cv2.warpPerspective(img_src, psp_matrix_P, (200,200))
            img_psp_M = cv2.warpPerspective(img_src, psp_matrix_M, (200,200))
            if modality == 'mask':
                cv2.imwrite(os.path.join(dirpath+'_HSR', filename), cv2.threshold(img_psp_P, 127, 255, cv2.THRESH_BINARY)[1])
                cv2.imwrite(os.path.join(dirpath+'_HSL', filename), cv2.threshold(img_psp_M, 127, 255, cv2.THRESH_BINARY)[1])
            else:
                cv2.imwrite(os.path.join(dirpath+'_HSR', filename), img_psp_P)
                cv2.imwrite(os.path.join(dirpath+'_HSL', filename), img_psp_M)

            # Rotation
            rot_matrix_P5 = cv2.getRotationMatrix2D((100,100), rot_angle, 1.0)
            rot_matrix_M5 = cv2.getRotationMatrix2D((100,100), -rot_angle, 1.0)
            img_rot_P5 = cv2.warpAffine(img_src, rot_matrix_P5, (200,200), flags=cv2.INTER_CUBIC)
            img_rot_M5 = cv2.warpAffine(img_src, rot_matrix_M5, (200,200), flags=cv2.INTER_CUBIC)
            if modality == 'mask':
                cv2.imwrite(os.path.join(dirpath+'_RTP', filename), cv2.threshold(img_rot_P5, 127, 255, cv2.THRESH_BINARY)[1])
                cv2.imwrite(os.path.join(dirpath+'_RTM', filename), cv2.threshold(img_rot_M5, 127, 255, cv2.THRESH_BINARY)[1])
            else:
                cv2.imwrite(os.path.join(dirpath+'_RTP', filename), img_rot_P5)
                cv2.imwrite(os.path.join(dirpath+'_RTM', filename), img_rot_M5)




# Main
if __name__ == '__main__':
    generateDeformedImage(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])



