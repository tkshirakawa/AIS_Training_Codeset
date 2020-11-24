'''
    Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com
    
    
    Released under the MIT license.
    https://opensource.org/licenses/mit-license.php

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


import os
import sys
import glob
import csv


print('Description : Make a CSV list of paths for image files in <case**> directories.')

if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a directory containing <case**> directories.')
    print('  argv[2] : Image file type, .jpg or .png.')
    sys.exit(0)


dirNameListX = ['x_A', 'x_A_CP0', 'x_A_CP1', 'x_A_CP2', 'x_A_CP3', 'x_A_CP4', 'x_A_DKM', 'x_A_DKP', 'x_A_HSR', 'x_A_HSL', 'x_A_HT', 'x_A_RTM', 'x_A_RTP', 'x_A_WD',
                'x_B', 'x_B_CP0', 'x_B_CP1', 'x_B_CP2', 'x_B_CP3', 'x_B_CP4', 'x_B_DKM', 'x_B_DKP', 'x_B_HSR', 'x_B_HSL', 'x_B_HT', 'x_B_RTM', 'x_B_RTP', 'x_B_WD',
                'x_C', 'x_C_CP0', 'x_C_CP1', 'x_C_CP2', 'x_C_CP3', 'x_C_CP4', 'x_C_DKM', 'x_C_DKP', 'x_C_HSR', 'x_C_HSL', 'x_C_HT', 'x_C_RTM', 'x_C_RTP', 'x_C_WD',
                # 'x_D', 'x_D_CP0', 'x_D_CP1', 'x_D_CP2', 'x_D_CP3', 'x_D_CP4', 'x_D_DKM', 'x_D_DKP', 'x_D_HSR', 'x_D_HSL', 'x_D_HT', 'x_D_RTM', 'x_D_RTP', 'x_D_WD',
                'x_O', 'x_O_CP0', 'x_O_CP1', 'x_O_CP2', 'x_O_CP3', 'x_O_CP4', 'x_O_DKM', 'x_O_DKP', 'x_O_HSR', 'x_O_HSL', 'x_O_HT', 'x_O_RTM', 'x_O_RTP', 'x_O_WD' ]
dirNameListY = ['y_O', 'y_O_CP0', 'y_O_CP1', 'y_O_CP2', 'y_O_CP3', 'y_O_CP4', 'y_O_DKM', 'y_O_DKP', 'y_O_HSR', 'y_O_HSL', 'y_O_HT', 'y_O_RTM', 'y_O_RTP', 'y_O_WD',
                'y_O', 'y_O_CP0', 'y_O_CP1', 'y_O_CP2', 'y_O_CP3', 'y_O_CP4', 'y_O_DKM', 'y_O_DKP', 'y_O_HSR', 'y_O_HSL', 'y_O_HT', 'y_O_RTM', 'y_O_RTP', 'y_O_WD',
                'y_O', 'y_O_CP0', 'y_O_CP1', 'y_O_CP2', 'y_O_CP3', 'y_O_CP4', 'y_O_DKM', 'y_O_DKP', 'y_O_HSR', 'y_O_HSL', 'y_O_HT', 'y_O_RTM', 'y_O_RTP', 'y_O_WD',
                # 'y_O', 'y_O_CP0', 'y_O_CP1', 'y_O_CP2', 'y_O_CP3', 'y_O_CP4', 'y_O_DKM', 'y_O_DKP', 'y_O_HSR', 'y_O_HSL', 'y_O_HT', 'y_O_RTM', 'y_O_RTP', 'y_O_WD',
                'y_O', 'y_O_CP0', 'y_O_CP1', 'y_O_CP2', 'y_O_CP3', 'y_O_CP4', 'y_O_DKM', 'y_O_DKP', 'y_O_HSR', 'y_O_HSL', 'y_O_HT', 'y_O_RTM', 'y_O_RTP', 'y_O_WD' ]


csvFile = open(os.path.join(sys.argv[1], 'MakeCSVList_result.csv'), 'w', newline='', encoding='utf-8')
csvWriter = csv.writer(csvFile)
csvWriter.writerow(['x', 'y'])


for X, Y in zip(dirNameListX, dirNameListY):
    fileListX = glob.glob(os.path.join(sys.argv[1], 'case*', X, '[0-9][0-9][0-9][0-9]'+sys.argv[2]))
    fileListY = glob.glob(os.path.join(sys.argv[1], 'case*', Y, '[0-9][0-9][0-9][0-9]'+sys.argv[2]))
    # fileListX = glob.glob(os.path.join(sys.argv[1], 'case[0-9][0-9]*', X, '[0-9][0-9][0-9][0-9]'+sys.argv[2]))
    # fileListY = glob.glob(os.path.join(sys.argv[1], 'case[0-9][0-9]*', Y, '[0-9][0-9][0-9][0-9]'+sys.argv[2]))

    if len(fileListX) <= 0 or len(fileListY) <= 0:
        print('> Directories : ' + X + ', ' + Y + ' ### EMPTY')
        continue
    elif len(fileListX) != len(fileListY):
        print('> Directories : ' + X + ', ' + Y + ' ### UNMATCH number of files')
        continue
    else:
        print('> Directories : ' + X + ', ' + Y)

    listData = []
    for pX, pY in zip(fileListX, fileListY):
        if os.path.basename(pX) == os.path.basename(pY):
            listData.append([pX, pY])
        else:
            print('### ERROR : File names of x and y are different!')
            print('> Path for x : ' + pX)
            print('> Path for y : ' + pY)

    csvWriter.writerows(listData)

csvFile.close()



