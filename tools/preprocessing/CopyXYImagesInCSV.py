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

