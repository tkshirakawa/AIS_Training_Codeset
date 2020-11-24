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
import shutil




if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Image file type, .jpg or .png.')
    print('  argv[2] : Path to a directory of input images.')
    print('  argv[3] : [Not used]')
    print('  argv[4] : Images or mask images? Input image or mask.')
    print('  argv[5] : Trimming width; positive value = top&bottom / negative value = right&left.')
    sys.exit(0)




def makeDirsByDeletingOld(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path)




def applyNlMeansDenoising(image, size=2, templateWindowSize=7, searchWindowSize=21):
    return cv2.fastNlMeansDenoising(image, None, size, templateWindowSize, searchWindowSize)




# def applyBilateralFilter(image, size=5, sigmaColor=10, sigmaSpace=10):
def applyBilateralFilter(image, size=-1, sigmaColor=3.0, sigmaSpace=1.5):
    image_tmp = image
    for i in range(3):
        image_tmp = cv2.bilateralFilter(image_tmp, size, sigmaColor, sigmaSpace)
    return image_tmp




def applyCLAHE(image, clipLimit=1.4, tileGridSize=(4,4)):
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=tileGridSize)
    return clahe.apply(image)




def applyPosterization(image, nstep=32):
    look_up_table = np.zeros((256, 1), dtype = 'uint8')
    step, mod = divmod(256, nstep)
    step_0 = step + mod // 2
    step_1 = 256 - step_0 - step * (nstep - 2)
    look_up_table[0:step_0][0] = step_0 / 2
    for istep in range(step_0, 256-step_1, step):
        iistep = min(istep + step, 256)
        look_up_table[istep:iistep][0] = (istep + iistep) / 2
    look_up_table[256-step_1:][0] = (step_1 + 255) / 2
    return cv2.LUT(cv2.bilateralFilter(image, 10, 10, 10), look_up_table)




###################################################################3
def generateDeformedImage(format_ext, data_dir, dummy, modality, trimming):

    black = [0]

    if format_ext != '.jpg' and format_ext != '.png':
        print('### ERROR: Use jpg or png for input and result images!')
        sys.exit(0)

    if modality == 'image':         # [ Original, A, B, C ]
        outputDirpath = [data_dir+'_O', data_dir+'_A', data_dir+'_B', data_dir+'_C']
    elif modality == 'mask':        # [ Original ]
        outputDirpath = [data_dir+'_O']
    else:
        print('### ERROR: Use image or mask!')
        sys.exit(0)


    files = glob.glob(os.path.join(data_dir, '[0-9][0-9][0-9][0-9]'+format_ext))
    if len(files) <= 0:
        print('### ERROR: No input files found!')
        sys.exit(0)


    # Image size
    tempImg = cv2.imread(files[0], cv2.IMREAD_UNCHANGED)    # [height, width, channel] for color images, [height, width] for grayscale images
    if tempImg.ndim != 2:
        print('### ERROR: Grayscale images required!')
        sys.exit(0)

    height, width = tempImg.shape
    if height != width:
        print('### ERROR: Square images required!')
        sys.exit(0)
    else:
        img_size = height   # = width


    # Parameters
    # For heart structure
    HW_padding = int(img_size * 0.050)          # For height and width deformation
    trz_padding = int(img_size * 0.125)         # For trapezoid deformation
    plg_padding = int(img_size * 0.100)         # For parallelogram deformation
    rot_angle = 8                               # For rotation

    # For AS calcium
    # HW_padding = int(img_size * 0.050)          # For height and width deformation
    # trz_padding = int(img_size * 0.150)         # For trapezoid deformation
    # plg_padding = int(img_size * 0.150)         # For parallelogram deformation
    # rot_angle = 15                              # For rotation


    clipTB = int(trimming)
    clipLR = 0
    if clipTB < 0:
        clipLR = -clipTB
        clipTB = 0


    for dirpath in outputDirpath:
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
        img_src_original = cv2.imread(fpath, cv2.IMREAD_UNCHANGED)
        img_src_tmp = img_src_original[clipTB:img_size-clipTB, clipLR:img_size-clipLR]
        print('Input : ' + fpath)

        for dirpath in outputDirpath:

            # Create color-changed images
            if dirpath[-1] == 'O':      # Original
                img_src = cv2.copyMakeBorder(img_src_tmp,
                                                clipTB, clipTB, clipLR, clipLR, cv2.BORDER_CONSTANT, value=black)
            elif dirpath[-1] == 'A':    # NlMeansDenoising
                img_src = cv2.copyMakeBorder(applyNlMeansDenoising(img_src_tmp),
                                                clipTB, clipTB, clipLR, clipLR, cv2.BORDER_CONSTANT, value=black)
            elif dirpath[-1] == 'B':    # bilateralFilter
                img_src = cv2.copyMakeBorder(applyBilateralFilter(img_src_tmp),
                                                clipTB, clipTB, clipLR, clipLR, cv2.BORDER_CONSTANT, value=black)
            elif dirpath[-1] == 'C':    # CLAHE
                img_src = cv2.copyMakeBorder(applyCLAHE(img_src_tmp),
                                                clipTB, clipTB, clipLR, clipLR, cv2.BORDER_CONSTANT, value=black)
            # elif dirpath[-1] == 'D':    # posterization
            #     img_src = cv2.copyMakeBorder(applyPosterization(img_src_tmp),
            #                                     clipTB, clipTB, clipLR, clipLR, cv2.BORDER_CONSTANT, value=black)
            # elif dirpath[-1] == 'Z':    # CLAHE+bilateralFilter
            #     img_src = cv2.copyMakeBorder(applyBilateralFilter(applyCLAHE(img_src_tmp)),
            #                                     clipTB, clipTB, clipLR, clipLR, cv2.BORDER_CONSTANT, value=black)

            cv2.imwrite(os.path.join(dirpath, filename), img_src)

            # Height / width
            img_ht = cv2.resize(img_src, dsize=(img_size,img_size+2*HW_padding), interpolation=cv2.INTER_CUBIC)
            img_wd = cv2.resize(img_src, dsize=(img_size+2*HW_padding,img_size), interpolation=cv2.INTER_CUBIC)
            if modality == 'mask':
                cv2.imwrite(os.path.join(dirpath+'_HT', filename), cv2.threshold(img_ht[HW_padding:img_size+HW_padding,0:img_size], 127, 255, cv2.THRESH_BINARY)[1])
                cv2.imwrite(os.path.join(dirpath+'_WD', filename), cv2.threshold(img_wd[0:img_size,HW_padding:img_size+HW_padding], 127, 255, cv2.THRESH_BINARY)[1])
            else:
                cv2.imwrite(os.path.join(dirpath+'_HT', filename), img_ht[HW_padding:img_size+HW_padding,0:img_size])
                cv2.imwrite(os.path.join(dirpath+'_WD', filename), img_wd[0:img_size,HW_padding:img_size+HW_padding])

            # Trapezoid
            perspective1 = np.float32([[0, img_size], [img_size, img_size], [img_size, 0], [0, 0]])
            perspective2_P = np.float32([[-trz_padding, img_size], [img_size+trz_padding, img_size], [img_size, 0], [0, 0]])
            perspective2_M = np.float32([[0, img_size], [img_size, img_size], [img_size+trz_padding, 0], [-trz_padding, 0]])
            psp_matrix_P = cv2.getPerspectiveTransform(perspective1, perspective2_P)
            psp_matrix_M = cv2.getPerspectiveTransform(perspective1, perspective2_M)
            img_psp_P = cv2.warpPerspective(img_src, psp_matrix_P, (img_size,img_size))
            img_psp_M = cv2.warpPerspective(img_src, psp_matrix_M, (img_size,img_size))
            if modality == 'mask':
                cv2.imwrite(os.path.join(dirpath+'_DKP', filename), cv2.threshold(img_psp_P, 127, 255, cv2.THRESH_BINARY)[1])
                cv2.imwrite(os.path.join(dirpath+'_DKM', filename), cv2.threshold(img_psp_M, 127, 255, cv2.THRESH_BINARY)[1])
            else:
                cv2.imwrite(os.path.join(dirpath+'_DKP', filename), img_psp_P)
                cv2.imwrite(os.path.join(dirpath+'_DKM', filename), img_psp_M)

            # Parallelogram
            perspective1 = np.float32([[0, img_size], [img_size, img_size], [img_size, 0], [0, 0]])
            perspective2_P = np.float32([[-plg_padding, img_size], [img_size, img_size], [img_size+plg_padding, 0], [0, 0]])
            perspective2_M = np.float32([[0, img_size], [img_size+plg_padding, img_size], [img_size, 0], [-plg_padding, 0]])
            psp_matrix_P = cv2.getPerspectiveTransform(perspective1, perspective2_P)
            psp_matrix_M = cv2.getPerspectiveTransform(perspective1, perspective2_M)
            img_psp_P = cv2.warpPerspective(img_src, psp_matrix_P, (img_size,img_size))
            img_psp_M = cv2.warpPerspective(img_src, psp_matrix_M, (img_size,img_size))
            if modality == 'mask':
                cv2.imwrite(os.path.join(dirpath+'_HSR', filename), cv2.threshold(img_psp_P, 127, 255, cv2.THRESH_BINARY)[1])
                cv2.imwrite(os.path.join(dirpath+'_HSL', filename), cv2.threshold(img_psp_M, 127, 255, cv2.THRESH_BINARY)[1])
            else:
                cv2.imwrite(os.path.join(dirpath+'_HSR', filename), img_psp_P)
                cv2.imwrite(os.path.join(dirpath+'_HSL', filename), img_psp_M)

            # Rotation
            rot_matrix_P = cv2.getRotationMatrix2D((img_size/2,img_size/2), rot_angle, 1.0)
            rot_matrix_M = cv2.getRotationMatrix2D((img_size/2,img_size/2), -rot_angle, 1.0)
            img_rot_P = cv2.warpAffine(img_src, rot_matrix_P, (img_size,img_size), flags=cv2.INTER_CUBIC)
            img_rot_M = cv2.warpAffine(img_src, rot_matrix_M, (img_size,img_size), flags=cv2.INTER_CUBIC)
            if modality == 'mask':
                cv2.imwrite(os.path.join(dirpath+'_RTP', filename), cv2.threshold(img_rot_P, 127, 255, cv2.THRESH_BINARY)[1])
                cv2.imwrite(os.path.join(dirpath+'_RTM', filename), cv2.threshold(img_rot_M, 127, 255, cv2.THRESH_BINARY)[1])
            else:
                cv2.imwrite(os.path.join(dirpath+'_RTP', filename), img_rot_P)
                cv2.imwrite(os.path.join(dirpath+'_RTM', filename), img_rot_M)




# Main
if __name__ == '__main__':
    generateDeformedImage(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])



