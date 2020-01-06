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


import sys

if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a CSV/.h5 file for training image paths')
    print('  argv[2] : Path to a CSV/.h5 file for validation image paths')
    print('  argv[3] : Path to a directory to save results in it')
    print('  argv[4] : Training mode: 0=Normal, 1=Resume the following model, 2=Boost the following model weights')
    print('  argv[5] : Path to a model for Mode-1 or Mode-2')
    print('  argv[6] : Initial epoch to resume training for Mode-1')
    print('  NOTE : Input images must be 200x200 gray-scale without alpha values')
    sys.exit()


import os
import warnings
warnings.filterwarnings('ignore')

import shutil
import time
import math
import platform
from datetime import datetime, timedelta, timezone

import tensorflow as tf
import keras.backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, LearningRateScheduler, Callback
from keras.utils import plot_model
from keras.models import load_model
import matplotlib.pyplot as plt

# For calculation with 16-bit float
# K.set_floatx('float16')
# K.set_epsilon(1e-4)     # default is 1e-7 which is too small for float16. Without adjusting the epsilon, we will get NaN predictions because of divide by zero problems.

# For PlaidML
#import plaidml.keras
#plaidml.keras.install_backend()

# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.95
# config.gpu_options.allow_growth = True
# sess = tf.Session(config=config)
# K.set_session(sess)




# Neural network model and parameters
#####################################################################

# Select neural networks
# from neural_networks import CV_net as NN
from neural_networks import CV_net_2 as NN
# from neural_networks import U_net as NN
# from neural_networks import Deeplab_v3_plus as NN

# Select optimizer
from keras.optimizers import Adam as OPTIMIZER
# from F16_func.AdamF16 import AdamF16 as OPTIMIZER

# Select loss function
# from utils.Validation_func import mean_iou_rou_MSE_loss as LOSS    ### Best performance ###
# from utils.Validation_func import mean_iou_rou2_MSE_loss as LOSS   ### Good ###
# from utils.Validation_func import mean_iou_loss as LOSS            ### Not bad ###
from utils.Validation_func import mean_iou_MSE_loss as LOSS        ### Best performance ###
# from utils.Validation_func import mean_iou_MAE_loss as LOSS        ### Good ###
# from utils.Validation_func import mean_iou_SVM_loss as LOSS
# from utils.Validation_func import mean_iou_SSVM_loss as LOSS
# from utils.Validation_func import mean_iou_LGC_loss as LOSS
# from utils.Validation_func import dice_coef_loss as LOSS           ### Not bad ###
# from utils.Validation_func import dice_coef_MSE_loss as LOSS       ### Best performance ###
# from utils.Validation_func import dice_coef_MAE_loss as LOSS       ### Good ###
# from utils.Validation_func import dice_coef_SVM_loss as LOSS
# from utils.Validation_func import dice_coef_SSVM_loss as LOSS
# from utils.Validation_func import dice_coef_LGC_loss as LOSS
# from keras.losses import mean_squared_error as LOSS                ### Not good ###
# from keras.losses import mean_absolute_error as LOSS               ### Not good ###

# Metrics indicated in each epoch
# NOTE: the following <Monitor for saving the best result> must be included in this custom_metrics
from utils.Validation_func import mean_iou, mean_rou, mean_iou_rou, mean_iou_rou2, dice_coef
custom_metrics = {'mean_iou': mean_iou, 'mean_rou': mean_rou, 'mean_iou_rou': mean_iou_rou, 'dice_coef': dice_coef}

# Monitor for saving the best result
pram_monitor = ['val_mean_iou', 'max']
# pram_monitor = ['val_mean_iou_rou', 'max']

# Batch size
# 16 for CV_net/CV\net2, 8 for U_net and Deeplab_v3_plus
# pram_batch_size = 8
pram_batch_size = 16

# Define a learning rate at a point of epoch by 'pram_LR_points = [[epoch, learning rate], ...]'
# pram_LR_points   = [[0, 7.81e-4], [5, 7.81e-4], [15, 6e-4], [30, 2e-4], [35, 1e-4], [50, 2e-5]]     # For aorta
# pram_LR_points  = [[0, 7.81e-4], [30, 7.81e-4], [50, 6e-4], [100, 2e-4], [140, 1e-4], [200, 1e-5]]     # For heart
# pram_LR_points   = [[0,2e-3], [50,2e-3], [80,1.5e-3], [100,1.1e-3], [150,3e-4], [200,1e-4]]     # For heart 20191116
pram_LR_points   = [[0,3e-3], [3,3.2e-3], [12,4.5e-3], [30,4.5e-3], [50,3e-3], [80,1e-3], [100,5e-4], [150,2e-4]]                     # For heart 20200106
# pram_LR_points   = [[0,3e-3], [3,3.2e-3], [12,4.8e-3], [30,4.8e-3], [50,3.8e-3], [80,2e-3], [100,1.1e-3], [150,3e-4], [200,1e-4]]     # For heart 20190903
# pram_LR_points   = [[0,3e-3], [3,3.2e-3], [12,4.8e-3], [30,4.8e-3], [50,3.8e-3], [80,2e-3], [100,1.1e-3], [180,3e-4], [250,6e-5]]     # For heart 20190903
# pram_LR_points   = [[0,3e-3], [3,3.2e-3], [12,4.8e-3], [30,4.8e-3], [50,3.8e-3], [80,2e-3], [120,1.1e-3], [180,3e-4], [250,1e-4]]       # For heart 20191107
# pram_LR_points   = [[0,1e-3], [100,1e-3]]       # For heart 20191118 Paper
#pram_LR_points   = [[0, 1.5e-4], [30, 1.0e-4], [80, 0.2e-4], [130, 0.05e-4], [180, 0.01e-4]]     # For heart
# pram_LR_points   = [[0, 7.81e-4], [2, 7.81e-4], [10, 2e-5]]     # For bone
pram_LR_step     = [1.0, 1.0]       # Multiplying steps for 'pram_LR_mltplier' when [decreasing, increasing]
pram_LR_limit    = [0.125, 8.0]     # Limitation of 'pram_LR_mltplier' when [decreasing, increasing]
pram_LR_patience = [4, 4]           # Patience counts before changing 'pram_LR_mltplier' when [decreasing, increasing]
pram_LR_mltplier = 1.0              # Initial multiplying value for learning rate
pram_SP_patience = 500              # Patience before EarlyStopping()

# Epoch and other parameters
# Resume training
if sys.argv[4] == '1':
    pram_init_epoch = int(sys.argv[6]) - 1      # Initial epoch of train (starting from zero)
    pram_epochs     = pram_LR_points[-1][0]
    if pram_init_epoch < 0 or pram_init_epoch >= pram_epochs:
        pram_init_epoch = min(pram_epochs-1, max(0, pram_init_epoch))
        print('ALART : Initial epoch [{0}] is clipped between 0 and {1}'.format(sys.argv[6], pram_epochs-1))
    trainMode = 'Resume training'
    trainName = 'Resumed model'
    trainDescript = 'Resume training'
    trainModelPath = sys.argv[5]
# Normal or Boost training
elif sys.argv[4] == '0' or sys.argv[4] == '2':
    pram_init_epoch = pram_LR_points[0][0]     # Initial epoch of train (starting from zero)
    pram_epochs     = pram_LR_points[-1][0]
    trainName = NN.Model_Name()
    trainDescript = NN.Model_Description()
    if sys.argv[4] == '0':
        trainMode = 'Normal training'
        trainModelPath = 'none'
    else:
        trainMode = 'Boost training'
        trainModelPath = sys.argv[5]
else:
    print('ERROR : Invalid training mode!!!')
    sys.exit()


# Define callbacks for learning rate
#####################################################################

class SetLRMultiplier(Callback):

    monitor_max = 0.0
    n_good = 0
    n_bad = 0

    def on_epoch_end(self, epoch, logs={}):
        global pram_LR_mltplier
        global pram_LR_step
        global pram_LR_limit
        global pram_LR_patience

        val_mont = logs.get(pram_monitor[0])

        if pram_LR_step[0] < 1.0 and val_mont < self.monitor_max:     # Decrease pram_LR_mltplier
            self.n_good = 0
            self.n_bad += 1
            if pram_LR_mltplier <= 1.0: n_patience = pram_LR_patience[0]                    # 8
            else:                       n_patience = math.ceil(pram_LR_patience[0] / 2)     # 4
            print('Counts of epochs with unimproved result: {0} /{1}'.format(self.n_bad, n_patience))
            if self.n_bad >= n_patience:
                pram_LR_mltplier = max(pram_LR_limit[0], pram_LR_mltplier*pram_LR_step[0])
                print('LR multiplier is set to {0} for the next epoch (step {1}, min {2})'.format(pram_LR_mltplier, pram_LR_step[0], pram_LR_limit[0]))
                # if pram_LR_mltplier*pram_LR_step[0] >= pram_LR_limit[0]:
                #     pram_LR_mltplier *= pram_LR_step[0]
                #     print('LR multiplier is set to {0} for the next epoch (step {1}, min {2})'.format(pram_LR_mltplier, pram_LR_step[0], pram_LR_limit[0]))
                # else:
                #     pram_LR_mltplier = 1.0
                #     print('LR multiplier is reset to 1.0 for the next epoch (step {0}, min {1})'.format(pram_LR_step[0], pram_LR_limit[0]))
                self.n_bad = 0
        elif pram_LR_step[1] > 1.0 and val_mont >= self.monitor_max:    # Increase pram_LR_mltplier
            self.monitor_max = val_mont
            self.n_good += 1
            self.n_bad = 0
            if pram_LR_mltplier >= 1.0: n_patience = pram_LR_patience[1]                    # 4
            else:                       n_patience = math.ceil(pram_LR_patience[1] / 2)     # 2
            print('Counts of epochs with improved result: {0} /{1}'.format(self.n_good, n_patience))
            if self.n_good >= n_patience:
                pram_LR_mltplier = min(pram_LR_limit[1], pram_LR_mltplier*pram_LR_step[1])
                print('LR multiplier is set to {0} for the next epoch (step {1}, max {2})'.format(pram_LR_mltplier, pram_LR_step[1], pram_LR_limit[1]))
                self.n_good = 0

        print('End of Epoch\n')


def CalcLearningRate(epoch):
    global pram_LR_mltplier
    global pram_LR_points

    def LR_at_epoch(epoch, pt1, pt2):
        return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0]) * (epoch - pt1[0]) + pt1[1]

    for i in range(len(pram_LR_points)-1):
        if pram_LR_points[i][0] <= epoch and epoch < pram_LR_points[i+1][0]:
            x = LR_at_epoch(epoch, pram_LR_points[i], pram_LR_points[i+1])
            break
    print('Learning rate is set to {0}, calculated from raw LR={1} and multiplier={2}'.format(pram_LR_mltplier*x, x, pram_LR_mltplier))

    return pram_LR_mltplier * x


# Model compile and learning
#####################################################################

JST = timezone(timedelta(hours=+9), 'JST')      # Japan Standard Time, Change for your time
startdate = datetime.now(JST)
starttime = time.time()

# Loaded neural network code may not have Custom_Layers()
try:    custom_layers = NN.Custom_Layers()
except: custom_layers = {}      # Empty

# Training mode: 0=Normal, 1=Resume the model, 2=Boost the model weights
if sys.argv[4] == '1':
    custom_loss = {LOSS.__name__: LOSS}
    model = load_model(trainModelPath, custom_objects=dict(**custom_loss, **custom_metrics, **custom_layers))
else:
    model = NN.Build_Model()
    if sys.argv[4] == '2':
        model.load_weights(trainModelPath)
    model.compile(optimizer          = OPTIMIZER(lr=pram_LR_points[0][1], beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False),
                  loss               = LOSS,                             # Custom loss
                  metrics            = list(custom_metrics.values()),    # Custom metrics
                  loss_weights       = None,
                  sample_weight_mode = None,
                  weighted_metrics   = None,
                  target_tensors     = None )

model.summary()
print('Date                    : {0}'.format(startdate))
print('TensorFlow version      : {0}'.format(tf.VERSION))
print('Keras version           : {0}'.format(tf.keras.__version__))
print('OS-version              : {0}'.format(platform.platform()))
print('Processor               : {0}'.format(platform.processor()))
print('__________________________________________________________________________________________________')
print('Training mode           : {0}'.format(trainMode))
print('Model name              : {0}'.format(trainName))
print('Model description       : {0}'.format(trainDescript))
print('Loaded model path       : {0}'.format(trainModelPath))
print('__________________________________________________________________________________________________')
print('Loss                    : {0}'.format(LOSS.__name__))
print('Metrics                 : {0}'.format(model.metrics_names[1:]))
print('Monitor for best        : {0}'.format(pram_monitor))
print('Custom layers           : {0}'.format(list(custom_layers.keys()) ))
print('Batch size              : {0}'.format(pram_batch_size))
print('Epochs                  : {0} - {1}'.format(pram_init_epoch+1, pram_epochs))
print('Learning rates          : {0}'.format(pram_LR_points))
print('LR multiplier           : {0}'.format(pram_LR_mltplier))
print('LR step                 : {0}'.format(pram_LR_step))
print('LR limit                : {0}'.format(pram_LR_limit))
print('Patience for LR         : {0}'.format(pram_LR_patience))
print('Patience for early stop : {0}'.format(pram_SP_patience))
print('==================================================================================================')


# Checkpoint
# key = input('Continue? [y/n] : ')
# if key != 'y' and key != 'Y':
#     print('Exit...')
#     sys.exit()


# Paths and directories
datestr = startdate.strftime("%Y%m%d%H%M%S")
traindir_path = os.path.join(sys.argv[3], 'run'+datestr+' ('+trainMode+')')
codedir_path = os.path.join(traindir_path, 'code')
tmp_path = os.path.join(traindir_path,'tmp_model'+datestr)
os.mkdir(traindir_path)
os.mkdir(codedir_path)
shutil.copy2(__file__, os.path.join(codedir_path, os.path.basename(__file__)))
if sys.argv[4] != '1':
    shutil.copy2(NN.__file__, os.path.join(codedir_path, os.path.basename(NN.__file__)))


# Define callbacks
print('Defining callbacks...')
checkpointer = ModelCheckpoint(tmp_path, monitor=pram_monitor[0], verbose=1, save_best_only=True, mode=pram_monitor[1])
earlyStopper = EarlyStopping(monitor=pram_monitor[0], min_delta=0, patience=pram_SP_patience, verbose=1, mode=pram_monitor[1])
LRmultipliersetter = SetLRMultiplier()
scheduleLR = LearningRateScheduler(CalcLearningRate, verbose=0)
csvlogger = CSVLogger(os.path.join(traindir_path,'training_log.csv'), separator=',', append=False)
tensorboard = TensorBoard(traindir_path, histogram_freq=0, batch_size=pram_batch_size, write_graph=True)


# Data generator
from utils.Image_data_generator import ImageDataGenerator_CSV_with_Header, ImageDataGenerator_h5_Dataset

print('Loading images for training...')
ext = os.path.splitext(sys.argv[1])[1]
if   ext == '.csv' :  training_dataset = ImageDataGenerator_CSV_with_Header('Train data from CSV', sys.argv[1], shuffle=True)
elif ext == '.h5'  :  training_dataset = ImageDataGenerator_h5_Dataset('image_training', sys.argv[1])
else               :  sys.exit()
print('Loading images for validation...')
ext = os.path.splitext(sys.argv[2])[1]
if   ext == '.csv' :  validation_dataset = ImageDataGenerator_CSV_with_Header('Validation data from CSV', sys.argv[2], shuffle=True)
elif ext == '.h5'  :  validation_dataset = ImageDataGenerator_h5_Dataset('image_validation', sys.argv[2])
else               :  sys.exit()


# Save network figure and parameters
plot_model(model, to_file=os.path.join(traindir_path,'model_figure.png'), show_shapes=True, show_layer_names=False)
with open(os.path.join(traindir_path,'training_parameters.txt'), mode='w') as path_file:
    path_file.write('Date                    : {0}\n'.format(startdate))
    path_file.write('TensorFlow version      : {0}\n'.format(tf.VERSION))
    path_file.write('Keras version           : {0}\n'.format(tf.keras.__version__))
    path_file.write('OS-version              : {0}\n'.format(platform.platform()))
    path_file.write('Processor               : {0}\n\n'.format(platform.processor()))
    path_file.write('Training mode           : {0}\n'.format(trainMode))
    path_file.write('Model name              : {0}\n'.format(trainName))
    path_file.write('Model description       : {0}\n'.format(trainDescript))
    path_file.write('Loaded model path       : {0}\n\n'.format(trainModelPath))
    path_file.write('Training images         : {0} sets in {1}\n'.format(training_dataset.length(), sys.argv[1]))
    path_file.write('Validation images       : {0} sets in {1}\n\n'.format(validation_dataset.length(), sys.argv[2]))
    path_file.write('Loss                    : {0}\n'.format(LOSS.__name__))
    path_file.write('Metrics                 : {0}\n'.format(model.metrics_names[1:]))
    path_file.write('Monitor for best        : {0}\n'.format(pram_monitor))
    path_file.write('Custom layers           : {0}\n'.format(list(custom_layers.keys()) ))
    path_file.write('Batch size              : {0}\n'.format(pram_batch_size))
    path_file.write('Epochs                  : {0} - {1}\n'.format(pram_init_epoch+1, pram_epochs))
    path_file.write('Learning rates          : {0}\n'.format(pram_LR_points))
    path_file.write('LR multiplier           : {0}\n'.format(pram_LR_mltplier))
    path_file.write('LR step                 : {0}\n'.format(pram_LR_step))
    path_file.write('LR limit                : {0}\n'.format(pram_LR_limit))
    path_file.write('Patience for LR         : {0}\n'.format(pram_LR_patience))
    path_file.write('Patience for early stop : {0}\n\n'.format(pram_SP_patience))
    model.summary(print_fn=lambda x: path_file.write(x + '\n'))


# Train the model
print('Starting model lerning...')
results = model.fit_generator(training_dataset.flow(rescale=1.0/225.0, batch_size=pram_batch_size),
    steps_per_epoch         = training_dataset.length() // pram_batch_size,
    epochs                  = pram_epochs,
    verbose                 = 1,
    # callbacks               = [checkpointer, earlyStopper, LRmultipliersetter, scheduleLR, csvlogger, tensorboard],
    callbacks               = [checkpointer, earlyStopper, LRmultipliersetter, scheduleLR, csvlogger],
	validation_data         = validation_dataset.flow(rescale=1.0/225.0, batch_size=pram_batch_size),
    validation_steps        = validation_dataset.length() // pram_batch_size,
    max_queue_size          = 2,
    workers                 = 1,
    use_multiprocessing     = False,
    shuffle                 = False,
    initial_epoch           = pram_init_epoch )


# Show results
print('Saving training graph...')
his_loss = results.history[model.metrics_names[0]]
his_met1 = results.history[model.metrics_names[1]]
his_met2 = results.history[model.metrics_names[2]]
his_valloss = results.history['val_'+model.metrics_names[0]]
his_valmet1 = results.history['val_'+model.metrics_names[1]]
his_valmet2 = results.history['val_'+model.metrics_names[2]]
xlen = range(len(his_loss))

fig = plt.figure()
ax1 = fig.add_subplot(111)      # Loss
ax2 = ax1.twinx()

ax1.plot(xlen, his_loss, marker='.', color='salmon', label=LOSS.__name__)
ax1.plot(xlen, his_valloss, marker='.', color='red', label='val_'+LOSS.__name__)
ax2.plot(xlen, his_met1, marker='.', color='deepskyblue', label=model.metrics_names[1])
ax2.plot(xlen, his_valmet1, marker='.', color='blue', label='val_'+model.metrics_names[1])
ax2.plot(xlen, his_met2, marker='.', color='limegreen', label=model.metrics_names[2])
ax2.plot(xlen, his_valmet2, marker='.', color='green', label='val_'+model.metrics_names[2])

ax1.set_xlabel('Epoch')
ax1.set_ylabel(LOSS.__name__)
ax1.set_yscale("log")
ax1.set_ylim([0.001, 1.0])
ax2.set_ylabel('Metrics')
ax2.set_yscale("log")
ax2.set_ylim([0.8, 1.0])

h1, l1 = ax1.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax1.legend(h1+h2, l1+l2, loc='lower center')

plt.savefig(os.path.join(traindir_path,'training_graph.png'))
# plt.show()


# Save the trained model
print('Saving trained model...')
infostr = '{0}, {1}={2:.4f}, {3} by {4}.h5'.format(datestr, model.metrics_names[1], max(his_valmet1), trainMode, trainName)

shutil.copy(tmp_path, os.path.join(traindir_path, 'model'+infostr))
time.sleep(20)
os.remove(tmp_path)

# Save the model without optimizer
# model.save(os.path.join(traindir_path, 'model(wo optimizer)'+infostr), include_optimizer=False)


print('==================================================================================================')
print('Computation time        : {0}'.format(timedelta(seconds=time.time()-starttime)))
print('From the date           : {0}'.format(startdate))
print('==================================================================================================')

