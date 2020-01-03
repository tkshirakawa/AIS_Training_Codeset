'''
    Copyright (c) 2019, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com
    
    
    Released under the BSD license.
    URL: https://opensource.org/licenses/BSD-2-Clause
    
    Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
    
    1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
    2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
    
    THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
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

