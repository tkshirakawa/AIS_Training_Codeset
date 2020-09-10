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

if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a directory to save generated cross section files in it')
    print('  argv[2] : Path to a STL file (.stl) to be sliced')
    print('  argv[3] : Axis normal to the slice plane (x/y/z)')
    print('  argv[4] : Offset to start slicing')
    print('  argv[5] : Slice interval (if negative, slicing order will be inverted)')
    print('  argv[6] : f = flip, m = mirror, none = do not flip or mirror')
    sys.exit()

import shutil
import os
import trimesh
import numpy as np
from PIL import Image, ImageOps


#trimesh.util.attach_to_log()


IMG_SIZE = 200

axis = sys.argv[3]
if axis == 'x' or axis == 'X':
    PLANE_NOR = [1, 0, 0]
    COL_DEL = 0
elif axis == 'y' or axis == 'Y':
    PLANE_NOR = [0, 1, 0]
    COL_DEL = 1
elif axis == 'z' or axis == 'Z':
    PLANE_NOR = [0, 0, 1]
    COL_DEL = 2
else:
    print('Invalid axis character !!!')
    sys.exit()


### for TEST
# Define the 8 vertices of the cube
#verts00 = [ [-4, -1, -3],
#            [+2, -2, -3],
#            [+2, +1, -3],
#            [-4, +1, -3],
#            [-2, -1, +5],
#            [+2, -1, +5],
#            [+2, +1, +5],
#            [-2, +1, +5] ]
## Define the 12 triangles composing the cube
#faces00 = [ [0,3,1],
#            [1,3,2],
#            [0,4,7],
#            [0,7,3],
#            [4,5,6],
#            [4,6,7],
#            [5,1,2],
#            [5,2,6],
#            [2,3,6],
#            [3,7,6],
#            [0,1,5],
#            [0,5,4] ]
#
# Create the STL mesh
#stl_data = mesh.Mesh(np.zeros(faces00.shape[0], dtype=mesh.Mesh.dtype))
#for i, f in enumerate(faces00):
#    for j in range(3):
#        stl_data.vectors[i][j] = verts00[f[j],:]




# Load the STL file
print('Loading STL file...')

if not os.path.isfile(sys.argv[2]):
    print('The STL file dose NOT exist !!!')
    sys.exit()


#stl_data = trimesh.Trimesh(vertices=verts00, faces=faces00)
stl_data = trimesh.load_mesh(sys.argv[2])
stl_bounds = np.delete(stl_data.bounds, obj=COL_DEL, axis=1)
print('Bounds of STL model: {0}'.format(stl_data.bounds))


offset = float(sys.argv[4])
trans = [0.,0.,0.]
trans[COL_DEL] = stl_data.bounds[0][COL_DEL] + offset
stl_data.vertices -= trans
print('Transposition matrix: {0}'.format(trans))




# Cut the mesh at ...
print('Computing cross sections...')

interval = float(sys.argv[5])
print('Slice interval: {0}'.format(interval))
if abs(interval) < 0.001:
    print('Slice interval is too small !!!')
    sys.exit()


z_extents = stl_data.bounds[1][COL_DEL] - stl_data.bounds[0][COL_DEL]
z_levels  = np.arange(z_extents, step=abs(interval))

cross_sections_in_STL = stl_data.section_multiplane(plane_origin=(0,0,0), plane_normal=PLANE_NOR, heights=z_levels)


mask_size = max(stl_bounds[1,0]-stl_bounds[0,0], stl_bounds[1,1]-stl_bounds[0,1])
corners = np.array([[1e8,1e8], [-1e8,-1e8]])
for cross_section in cross_sections_in_STL:
    if cross_section is not None:
#        print(cross_section.bounds)
        corners[0,0] = min(corners[0,0], cross_section.bounds[0,0])
        corners[0,1] = min(corners[0,1], cross_section.bounds[0,1])
        corners[1,0] = max(corners[1,0], cross_section.bounds[1,0])
        corners[1,1] = max(corners[1,1], cross_section.bounds[1,1])
#print(stl_bounds, mask_size)
#print(corners)




# Rasterize cross section paths and save the mask files
print('Saving mask files...')

filename = os.path.splitext(os.path.basename(sys.argv[2]))[0]
dirpath = os.path.join(sys.argv[1], filename+' offset='+sys.argv[4]+' interval='+sys.argv[5]+' axis='+axis)
print('Dest. directory: ' + dirpath)
if os.path.exists(dirpath):
    key = input('!!! The directory exists. Overwrite? [y/n] : ')
    if key == 'y' or key == 'Y':
        shutil.rmtree(dirpath)
    else:
        print('Exit...')
        sys.exit()
os.makedirs(dirpath)


img_pitch = mask_size / IMG_SIZE
img_origin = (corners[0,0] - (mask_size - (corners[1,0] - corners[0,0])) / 2. , corners[0,1] - (mask_size - (corners[1,1] - corners[0,1])) / 2.)
img_size = (IMG_SIZE, IMG_SIZE)
img_counts = len(cross_sections_in_STL)
if interval > 0:
    istart = 1
    istep = 1
    print('Cross sections: {0}, with normal order'.format(img_counts))
else:
    istart = img_counts
    istep = -1
    print('Cross sections: {0}, with inverted order'.format(img_counts))


index = istart
for cross_section in cross_sections_in_STL:
    if cross_section is not None:   # When a cross section exists in this slicing plane
        r = cross_section.rasterize(pitch=img_pitch, origin=img_origin, resolution=img_size, fill=True, width=None)
        if sys.argv[6] == 'f' or sys.argv[6] == 'F':    result = ImageOps.flip(r.convert('L'))
        elif sys.argv[6] == 'm' or sys.argv[6] == 'M':  result = ImageOps.mirror(r.convert('L'))
        else:                                           result = r.convert('L')
    else:                           # When this slicing plane is empty
        result = Image.new('L', img_size, color=0)

    result.save(os.path.join(dirpath, 'slice{:0=4}.png'.format(index)))
    index += istep
#    mask.show()

#combined = np.sum(cross_sections_in_STL)
#combined.show()



