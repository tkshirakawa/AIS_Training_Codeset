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
import glob
import csv


if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a directory containing <case**> directories')
    print('  argv[2] : Image file type, .jpg or .png')
    sys.exit()


# dirNameListX = ['x', 'x_C', 'x_C_CP0', 'x_C_CP1', 'x_C_CP2', 'x_C_CP3', 'x_C_CP4', 'x_C_DKM', 'x_C_DKP', 'x_C_HSR', 'x_C_HSL', 'x_C_HT', 'x_C_RTM', 'x_C_RTP', 'x_C_WD',
#                      'x_O', 'x_O_CP0', 'x_O_CP1', 'x_O_CP2', 'x_O_CP3', 'x_O_CP4', 'x_O_DKM', 'x_O_DKP', 'x_O_HSR', 'x_O_HSL', 'x_O_HT', 'x_O_RTM', 'x_O_RTP', 'x_O_WD' ]
# dirNameListY = ['y', 'y_O', 'y_O_CP0', 'y_O_CP1', 'y_O_CP2', 'y_O_CP3', 'y_O_CP4', 'y_O_DKM', 'y_O_DKP', 'y_O_HSR', 'y_O_HSL', 'y_O_HT', 'y_O_RTM', 'y_O_RTP', 'y_O_WD',
#                      'y_O', 'y_O_CP0', 'y_O_CP1', 'y_O_CP2', 'y_O_CP3', 'y_O_CP4', 'y_O_DKM', 'y_O_DKP', 'y_O_HSR', 'y_O_HSL', 'y_O_HT', 'y_O_RTM', 'y_O_RTP', 'y_O_WD' ]
dirNameListX = ['x_C', 'x_C_CP0', 'x_C_CP1', 'x_C_CP2', 'x_C_CP3', 'x_C_CP4', 'x_C_DKM', 'x_C_DKP', 'x_C_HSR', 'x_C_HSL', 'x_C_HT', 'x_C_RTM', 'x_C_RTP', 'x_C_WD',
                'x_O', 'x_O_CP0', 'x_O_CP1', 'x_O_CP2', 'x_O_CP3', 'x_O_CP4', 'x_O_DKM', 'x_O_DKP', 'x_O_HSR', 'x_O_HSL', 'x_O_HT', 'x_O_RTM', 'x_O_RTP', 'x_O_WD' ]
dirNameListY = ['y_O', 'y_O_CP0', 'y_O_CP1', 'y_O_CP2', 'y_O_CP3', 'y_O_CP4', 'y_O_DKM', 'y_O_DKP', 'y_O_HSR', 'y_O_HSL', 'y_O_HT', 'y_O_RTM', 'y_O_RTP', 'y_O_WD',
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

