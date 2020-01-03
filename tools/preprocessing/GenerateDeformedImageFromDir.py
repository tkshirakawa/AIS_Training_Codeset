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
    print('  argv[1] : Image file type; Input .jpg or .png')
    print('  argv[2] : Path to a directory containing <case01>, <case02>, ,,, <case**> directories')
    print('  argv[3] : Trimming width; positive value = top&bottom / negative value = right&left')
    sys.exit()


fileListX = glob.glob(os.path.join(sys.argv[2], 'case*', 'x'))
fileListY = glob.glob(os.path.join(sys.argv[2], 'case*', 'y'))


if len(fileListX) <= 0 or len(fileListY) <= 0:
    print('> EMPTY cases')
    sys.exit()
elif len(fileListX) != len(fileListY):
    print('> UNMATCH number of cases')
    sys.exit()


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



