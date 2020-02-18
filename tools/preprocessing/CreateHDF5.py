'''
    Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com

    Released under the BSD 3-Clause License
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



def ReadImage_and_CreateH5(paths, dataname, shuffle=True):

    if paths is None:
        return

    csvlength = len(paths)

    print(' ')
    print('Data name : {0}'.format(dataname))
    print('Images    : {0}'.format(csvlength))


    # Store image data as 'uint8'
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




def Get_CSV_Paths(csvfile):

    if csvfile == 'none':
        return None

    # Open the CSV file
    with open(csvfile, 'r', newline='', encoding='utf-8_sig') as f:
        reader = csv.reader(f)
        next(reader)                                # !!! First row is a header, skip !!!
        paths = [row for row in reader]

    return paths




def Check_Duplication(list1, list2):

    if list1 is None or list2 is None:
        return

    import itertools

    tp = list(itertools.chain.from_iterable(training_paths))
    vp = list(itertools.chain.from_iterable(validation_paths))
    dup = list(set(tp) & set(vp))

    if len(dup) > 0:
        print('ERROR!!! Duplicated path(s) found in training and validation datasets !!!')
        print('Counts : {}'.format(len(dup)))
        for p in dup: print(p)
        sys.exit()
    else:
        print('Good! No duplicated path(s) found in training and validation datasets.')




# Read images and create h5 dataset files for training and validation
##########################################################################

print(' ')
print('Read images from CSV list and create h5 dataset for training and validation.')
print('Training CSV   : {0}'.format(sys.argv[1]))
print('Validation CSV : {0}'.format(sys.argv[2]))
print(' ')


training_paths   = Get_CSV_Paths(sys.argv[1])
validation_paths = Get_CSV_Paths(sys.argv[2])


print('Checking data duplication...')
Check_Duplication(training_paths, validation_paths)


ReadImage_and_CreateH5(training_paths, 'image_training', shuffle=True)
ReadImage_and_CreateH5(validation_paths, 'image_validation', shuffle=True)



