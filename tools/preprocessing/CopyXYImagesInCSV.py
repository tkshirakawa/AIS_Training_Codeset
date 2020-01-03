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
import shutil
import sys
import csv




print('Description : Copy images X (raw image) and Y (its mask) described in a source csv list to a destination directory.')

if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a CSV file for paths of source images')
    print('  argv[2] : Path to a destination directory to copy images in it')
    sys.exit()




print('Source CSV      : {0}'.format(sys.argv[1]))
print('Dest. directory : {0}'.format(sys.argv[2]))


with open(sys.argv[1], 'r', newline='') as f1:
    reader1 = csv.reader(f1)
    next(reader1)                                # !!! First row is a header, skip !!!
    paths = [row for row in reader1]

f2 = open(os.path.join(sys.argv[2], 'copied_files.csv'), 'w', newline='', encoding='utf-8')
writer2 = csv.writer(f2)
writer2.writerow(['file name', 'src_x', 'src_y'])


x_dir = os.path.join(sys.argv[2], 'x')
y_dir = os.path.join(sys.argv[2], 'y')
os.makedirs(x_dir, exist_ok=True)
os.makedirs(y_dir, exist_ok=True)

ifile = 0
listData = []
for p in paths:
    ifile += 1
    fnum = str(ifile).zfill(4)
    shutil.copyfile(p[0], os.path.join(x_dir, fnum+os.path.splitext(p[0])[1]))
    shutil.copyfile(p[1], os.path.join(y_dir, fnum+os.path.splitext(p[1])[1]))
    listData.append([fnum+'.ext', p[0], p[1]])

print('Image counts    : {0}'.format(ifile))


writer2.writerows(listData)
f2.close()

