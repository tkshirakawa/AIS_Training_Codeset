'''
    Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com
    
    
    Released under the MIT license.
    https://opensource.org/licenses/mit-license.php

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


# Import modules
import os
import sys
from datetime import datetime

from keras.models import load_model

# Import others
from Validation_func import mean_iou, dice_coef
from Validation_func import mean_iou_loss as LOSS
#from Validation_func import dice_coef_loss as LOSS
#from keras.losses import mean_squared_error as LOSS
#from keras.losses import mean_absolute_error as LOSS


# Show helps
if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a model to be RE-shaped')
    sys.exit()


# Load existing model
#model = load_model(sys.argv[1], custom_objects={'mean_iou': mean_iou})
model = load_model(sys.argv[1], custom_objects={LOSS.__name__: LOSS, 'mean_iou': mean_iou, 'dice_coef': dice_coef})

model.summary()
print('Loaded NN         : ', sys.argv[1])
print('Loss              : ', model.metrics_names[0])
print('Metrics           : ', model.metrics_names[1])
print('==================================================================================================')


# Save
datestr = datetime.now().strftime("%Y%m%d%H%M%S")
model.save('(reshaped '+datestr+') '+os.path.basename(sys.argv[1]), include_optimizer=False)

