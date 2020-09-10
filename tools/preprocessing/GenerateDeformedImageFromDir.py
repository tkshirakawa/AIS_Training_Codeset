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




if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Image file type, .jpg or .png.')
    print('  argv[2] : Path to a directory containing <case01>, <case02>, ,,, <case**> directories.')
    print('  argv[3] : Trimming width; positive value = top&bottom / negative value = right&left.')
    sys.exit(0)


fileListX = glob.glob(os.path.join(sys.argv[2], 'case*', 'x'))
fileListY = glob.glob(os.path.join(sys.argv[2], 'case*', 'y'))


if len(fileListX) <= 0 or len(fileListY) <= 0:
    print('> EMPTY cases')
    sys.exit(0)
elif len(fileListX) != len(fileListY):
    print('> UNMATCH number of cases')
    sys.exit(0)


for pX, pY in zip(fileListX, fileListY):
    print('------------------------------------------------------')
    print('> Directories for x and y...')
    print('> x : ' + pX)
    print('> y : ' + pY)

    if os.path.dirname(pX) == os.path.dirname(pY):
        from GenerateDeformedImage import generateDeformedImage
        print('> Calling generateDeformedImage...')
        generateDeformedImage(sys.argv[1], pX, 0, 'image', sys.argv[3])
        generateDeformedImage(sys.argv[1], pY, 0, 'mask', sys.argv[3])
    else:
        print('### ERROR : Case names of x and y are different!')



