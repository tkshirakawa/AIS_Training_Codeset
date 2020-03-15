'''
    Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com

    Released under the BSD 3-Clause License
'''


##### For TensorFlow v2.0 #####
# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function


import sys
import os

if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a neural network model file (.py).')
    print('  argv[2] : Path to a CSV/.h5 file for training image paths')
    print('  argv[3] : Path to a CSV/.h5 file for validation image paths')
    print('  argv[4] : Path to a directory to save results in it')
    print('  argv[5] : Training mode: 0=Normal, 1=Resume by load_model(), 2=Retrain by load_weights()')
    print('  argv[6] : Path to a model for mode 1 or 2')       # print('  argv[6] : Path to a model for Mode-1 or Mode-2')
    print('  argv[7] : Initial epoch to resume training for Mode-1')
    print('  NOTE : Input images must be 200x200 gray-scale without alpha values')
    sys.exit()




# The absolute path to this file and directory
exeFilePath = os.path.abspath(__file__)
exeDirPath = os.path.dirname(exeFilePath)
validFuncPath = os.path.join(exeDirPath, 'utils', 'Loss_and_metrics.py')

import importlib.machinery as imm


# Define loss and metrics
# get_loss() and get_metrics() will be used in coremltools when converting a model trained in this sequence

_LM = imm.SourceFileLoader('Loss_and_metrics', validFuncPath).load_module()

loss = _LM.mean_iou_MSE_loss            ### Best performance ###
# loss = _LM.mean_iou_MSE2_loss
# loss = _LM.mean_iou_BCE_loss
# loss = _LM.mean_iou_MSE_border_MSE_loss
# loss = _LM.mean_iou_MSE_border_BCE_loss
# loss = _LM.mean_iou_MSE_border_KLD_loss
# loss = _LM.mean_iou_SVM_loss
# loss = _LM.mean_iou_SSVM_loss
# loss = _LM.mean_iou_LGC_loss
# loss = _LM.mean_iou_rou_MSE_loss        ### Best performance ###
# loss = _LM.mean_rou
# loss = _LM.Huber_loss
# loss = _LM.mean_iou_Huber_loss
# loss = _LM.mean_moliou_loss
# loss = _LM.mean_moliou_MSE_loss
# loss = _LM.dice_coef_loss
# loss = _LM.dice_coef_MSE_loss           ### Best performance ###
# loss = _LM.dice_coef_SVM_loss
# loss = _LM.dice_coef_SSVM_loss
# loss = _LM.dice_coef_LGC_loss
# loss = _LM.cosine_distance_loss
# loss = _LM.mean_squared_error_loss      ### Not good ###

def get_loss(): return {loss.__name__: loss}

# Metrics
# NOTE: The metrics defined here must include the following LR_params['monitor_for_best'], e.g. 'mean_iou' here and 'val_mean_iou' in LR_params
def get_metrics(): return {'mean_iou': _LM.mean_iou, 'mean_rou': _LM.mean_rou, 'dice_coef': _LM.dice_coef}
#def get_metrics(): return {'mean_iou': _LM.mean_iou, 'mean_rou': _LM.mean_rou, 'mean_iou_rou': _LM.mean_iou_rou, 'dice_coef': _LM.dice_coef}




# Define a custom callback
#####################################################################

from keras.callbacks import Callback

class AutoLRManager(Callback):

    def __init__(self, param, early_stop, **kwargs):
        super(AutoLRManager, self).__init__(**kwargs)
        self.LR_decay = 1.0
        self.p = param
        self.early_stop = early_stop
        if self.p['monitor_for_best'][1] == 'max': self.best_val = -1e7
        else:                                      self.best_val = 1e7
        self.n_good = 0
        self.n_bad = 0

    def validation_monitor_improved(self, mont_val):
        if self.p['monitor_for_best'][1] == 'max':
            if mont_val >= self.best_val: return True
            else:                         return False
        else:
            if mont_val < self.best_val:  return True
            else:                         return False

    def on_epoch_end(self, epoch, logs=None):
        mont_val = logs.get(self.p['monitor_for_best'][0])
        print('{0} in this epoch = {1}'.format(self.p['monitor_for_best'][0], mont_val))

        if not self.validation_monitor_improved(mont_val):
            self.n_good = 0
            self.n_bad += 1
            step = self.p['step'][0]
            print('Epochs with UN-improved result: {0} /{1} [early stop {2}]'.format(self.n_bad, self.p['patience'][0], self.early_stop))
            if self.n_bad >= self.p['patience'][0] and step != 1.0:
                if   step < 1.0: self.LR_decay = max(self.p['limit'][0], self.LR_decay * step)
                elif step > 1.0: self.LR_decay = min(self.p['limit'][0], self.LR_decay * step)
                print('LR decay is set to {0} for the next epoch (step {1}, min {2})'.format(self.LR_decay, step, self.p['limit'][0]))
                self.n_bad = 0
        else:
            self.best_val = mont_val
            self.n_good += 1
            self.n_bad = 0
            step = self.p['step'][1]
            print('Epochs with improved result: {0} /{1}'.format(self.n_good, self.p['patience'][1]))
            if self.n_good >= self.p['patience'][1] and step != 1.0:
                if   step < 1.0: self.LR_decay = max(self.p['limit'][1], self.LR_decay * step)
                elif step > 1.0: self.LR_decay = min(self.p['limit'][1], self.LR_decay * step)
                print('LR decay is set to {0} for the next epoch (step {1}, max {2})'.format(self.LR_decay, step, self.p['limit'][1]))
                self.n_good = 0

        print('End of Epoch\n\n')
    
    def get_LR_decay(self):
        return self.LR_decay




# Main training code
#####################################################################

def Train():

    import shutil
    import time
    import platform
    import matplotlib.pyplot as plt
    from datetime import datetime, timedelta, timezone

    ##### For TensorFlow v2.0 #####
    # import tensorflow as tf
    # from tensorflow import keras
    # from tensorflow.keras import backend as K
    # # from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, LearningRateScheduler
    # from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
    # from tensorflow.keras.utils import plot_model
    # from tensorflow.keras.models import load_model

    import tensorflow as tf
    import keras.backend as K
    # from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, LearningRateScheduler
    from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, LearningRateScheduler
    from keras.utils import plot_model
    from keras.models import load_model


    # Initial parameters for learning rate (LR)
    # Those values can be overwritten afterwards or at anytime you like
    LR_params = {'formula'          : [None, 0.0, 0],               # Learning rate formula calculates LR at points of epochs - ['poly', base_lr, number_of_epochs] is available
                 'graph'            : [[0,4e-3], [100,2e-3]],       # Learning rate graph defines LR at points of epochs - [[epoch_1, LR_1], [epoch_2, LR_2], ... [epoch_last, LR_last]]
                 'step'             : [0.1, 2.0],                   # Multiplying values for LR decay - will be applied when monitor_for_best is [NOT improved, improved]
                 'limit'            : [0.01, 4.0],                  # Limitation of LR decay - when [NOT improved, improved]
                 'patience'         : [1000, 1000],                 # Patience counts before applying step for LR decay - when [NOT improved, improved]
                 'monitor_for_best' : ['val_mean_iou', 'max'] }     # Monitor for saving the best result


    # Load loss and metrics
    custom_loss = get_loss()
    custom_metrics = get_metrics()


    # Neural network model
    NN_model_path = sys.argv[1]
    NN = imm.SourceFileLoader(os.path.splitext(os.path.basename(NN_model_path))[0], NN_model_path).load_module()

    NN_info = []


    try:    NN_model_name    = NN.Model_Name()
    except: NN_model_name, _ = NN.__name__, NN_info.append('ALERT: Define a model name in the neural network model file.')

    try:    NN_model_descript    = NN.Model_Description()
    except: NN_model_descript, _ = 'Empty description.', NN_info.append('ALERT: Define description for the model in the neural network model file.')


    # Number of classes in the loaded neural network model
    try:    NN_num_classes    = NN.Number_of_Classes()
    except: NN_num_classes, _ = 1, NN_info.append('NOTE: The number of classes was not defined in the neural network model file, automatically set to 1.')
    _LM.set_num_classes(NN_num_classes)


    # Batch size
    # 16 for CV_net/CV\net2, 8 for U_net and Deeplab_v3_plus
    try:    NN_batch_size    = NN.Batch_Size()
    except: NN_batch_size, _ = 16, NN_info.append('NOTE: The batch size was not defined in the neural network model file, automatically set to 16.')


    # Define learning rates
    try:    LR_params['formula']    = NN.Learning_Rate_Formula()
    except: LR_params['formula'], _ = [None, 0.0, 0], \
                NN_info.append('ALERT: The LR formula was not defined in the neural network model file, automatically deactivated.')

    try:    LR_params['graph']    = NN.Learning_Rate_Lsit()
    except: LR_params['graph'], _ = [[0,3e-3], [3,3.2e-3], [12,4.5e-3], [30,4.5e-3], [50,3e-3], [80,1e-3], [100,5e-4], [150,2e-4]], \
                NN_info.append('ALERT: The LR graph was not defined in the neural network model file, automatically set.')
    # LR_params['graph'] = [[0, 7.81e-4], [5, 7.81e-4], [15, 6e-4], [30, 2e-4], [35, 1e-4], [50, 2e-5]]     # For aorta
    # LR_params['graph'] = [[0, 7.81e-4], [30, 7.81e-4], [50, 6e-4], [100, 2e-4], [140, 1e-4], [200, 1e-5]]     # For heart
    # LR_params['graph'] = [[0,2e-3], [50,2e-3], [80,1.5e-3], [100,1.1e-3], [150,3e-4], [200,1e-4]]     # For heart 20191116
    # LR_params['graph'] = [[0,3e-3], [3,3.2e-3], [12,4.5e-3], [30,4.5e-3], [50,3e-3], [80,1e-3], [100,5e-4], [150,2e-4]]                     # For heart 20200106
    # LR_params['graph'] = [[0,3e-3], [3,3.2e-3], [12,4.8e-3], [30,4.8e-3], [50,3.8e-3], [80,2e-3], [100,1.1e-3], [150,3e-4], [200,1e-4]]     # For heart 20190903
    # LR_params['graph'] = [[0,3e-3], [3,3.2e-3], [12,4.8e-3], [30,4.8e-3], [50,3.8e-3], [80,2e-3], [100,1.1e-3], [180,3e-4], [250,6e-5]]     # For heart 20190903
    # LR_params['graph'] = [[0,3e-3], [3,3.2e-3], [12,4.8e-3], [30,4.8e-3], [50,3.8e-3], [80,2e-3], [120,1.1e-3], [180,3e-4], [250,1e-4]]       # For heart 20191107
    # LR_params['graph'] = [[0,1e-3], [100,1e-3]]       # For heart 20191118 Paper
    # LR_params['graph'] = [[0, 1.5e-4], [30, 1.0e-4], [80, 0.2e-4], [130, 0.05e-4], [180, 0.01e-4]]     # For heart
    # LR_params['graph'] = [[0, 7.81e-4], [2, 7.81e-4], [10, 2e-5]]     # For bone

    if LR_params['formula'][0] is None: number_of_epochs = LR_params['graph'][-1][0]
    else:                               number_of_epochs = LR_params['formula'][2]


    # Patience before early stopping
    try:    patience_for_stop    = NN.Count_before_Stop()
    except: patience_for_stop, _ = 25, \
                NN_info.append('ALERT: The count for EarlyStopping() was not defined in the neural network model file, automatically set to 25.')


    # Resume training
    if sys.argv[5] == '1':
        init_epoch = int(sys.argv[7]) - 1      # Starting from zero
        if init_epoch < 0 or init_epoch >= number_of_epochs:
            init_epoch = min(number_of_epochs-1, max(0, init_epoch))
            print('ALART : Initial epoch [{0}] is clipped between 0 and {1}'.format(sys.argv[7], number_of_epochs-1))
        trained_model_path = sys.argv[6]
        training_mode = 'Resume training'
        NN_model_descript = 'The following model will be resumed - ' + trained_model_path + '\n\t\t\t' + NN_model_descript

    # Normal training or Retraining
    elif sys.argv[5] == '0' or sys.argv[5] == '2':
        init_epoch = 0       # Starting from zero
        if sys.argv[5] == '0':
            trained_model_path = None
            training_mode = 'Normal training'
        else:
            trained_model_path = sys.argv[6]
            training_mode = 'Retraining with trained wieghts'
            NN_model_descript = 'Trained weights will be loaded from - ' + trained_model_path + '\n\t\t\t' + NN_model_descript

    else:
        print('ERROR : Invalid training mode!!!')
        sys.exit()


    JST = timezone(timedelta(hours=+9), 'JST')      # Japan Standard Time, Change for your time
    startdate = datetime.now(JST)
    starttime = time.time()


    # Loaded neural network code may not have Custom_Layers()
    try:    custom_layers    = NN.Custom_Layers()
    except: custom_layers, _ = {}, \
                NN_info.append('ALERT: The dictionary of custom layers was not defined in the neural network model file, automatically set to empty.')


    # Training mode: 0=Normal, 1=Resume the model, 2=Boost the model weights
    if sys.argv[5] == '0':
        model = NN.Build_Model()
    elif sys.argv[5] == '1':
        model = load_model(trained_model_path, custom_objects=dict(**custom_loss, **custom_metrics, **custom_layers), compile=False)
    elif sys.argv[5] == '2':
        model = NN.Build_Model()
        model.load_weights(trained_model_path)
    else:
        print('Invalid mode.')
        sys.exit()


    # Optimizers
    # from tensorflow.keras.optimizers import SGD, Adam, Nadam
    from keras.optimizers import SGD, Adam, Nadam
    # from optimizers.AdaBound1.adabound1 import AdaBound1
    # from optimizers.AdaBound2.adabound2 import AdaBound2
    # from optimizers.Santa.Santa import Santa
    if LR_params['formula'][0] is not None: base_lr = LR_params['formula'][1]
    elif LR_params['graph'] is not None:    base_lr = LR_params['graph'][0][1]
    else:
        print('Invalid learning rate.')
        sys.exit()

    try:    optimizer    = NN.Optimizer(base_lr=base_lr)
    except: optimizer, _ = Adam(lr=base_lr, beta_1=0.9, beta_2=0.999, amsgrad=False), \
                NN_info.append('ALERT: The optimizer was not defined in the neural network model file, automatically set to Adam.')
    # except: optimizer, _ = SGD(lr=base_lr, momentum=0.9, nesterov=True), \
    # except: optimizer, _ = Nadam(lr=base_lr, beta_1=0.9, beta_2=0.999), \
    # except: optimizer, _ = AdaBound1(lr=base_lr, final_lr=0.5, beta_1=0.9, beta_2=0.999, gamma=1e-3, amsbound=False, weight_decay=0.0), \
    # except: optimizer, _ = AdaBound2(lr=base_lr, beta_1=0.9, beta_2=0.999, terminal_bound=0.5, lower_bound=0.0, upper_bound=None), \
    # except: optimizer, _ = Santa(lr=base_lr, exploration=max(number_of_epochs/2, number_of_epochs-20), rho=0.95, anne_rate=0.5), \


    # Compile
    model.compile(optimizer          = optimizer,
                  loss               = loss,
                  metrics            = list(custom_metrics.values()),
                  loss_weights       = None,
                  sample_weight_mode = None,
                  weighted_metrics   = None,
                  target_tensors     = None )


    # Paths and directories
    datestr = startdate.strftime("%Y%m%d%H%M%S")
    work_dir_path  = os.path.join(sys.argv[4], 'run'+datestr+' ('+training_mode+')')
    code_dir_path  = os.path.join(work_dir_path, 'code')
    NN_dir_path    = os.path.join(code_dir_path, 'neural_networks')
    utils_dir_path = os.path.join(code_dir_path, 'utils')
    tmp_model_path = os.path.join(work_dir_path, 'tmp_model'+datestr+'.h5')

    os.makedirs(NN_dir_path)
    os.makedirs(utils_dir_path)
    shutil.copy2(exeFilePath,   os.path.join(code_dir_path,  os.path.basename(exeFilePath)))        # Copy this file
    shutil.copy2(NN_model_path, os.path.join(NN_dir_path,    os.path.basename(NN_model_path)))      # Copy model file
    shutil.copy2(validFuncPath, os.path.join(utils_dir_path, os.path.basename(validFuncPath)))      # Copy loss and metrics file
    if custom_layers:
        shutil.copy2(os.path.join(exeDirPath, 'neural_networks', 'Custom_layers.py'), os.path.join(NN_dir_path, 'Custom_layers.py'))   # Copy layer file


    # Descriptions
    model.summary()
    print('Date                    : {0}'.format(startdate))
    print('TensorFlow version      : {0}'.format(tf.version.VERSION))
    print('Keras version           : {0}'.format(tf.keras.__version__))
    print('OS-version              : {0}'.format(platform.platform()))
    print('Processor               : {0}'.format(platform.processor()))
    print('__________________________________________________________________________________________________')
    print('Training mode           : {0}'.format(training_mode))
    print('Model name              : {0}'.format(NN_model_name))
    print('Model description       : {0}'.format(NN_model_descript))
    print('Number of classes       : {0}'.format(NN_num_classes))
    print('Loaded model path       : {0}'.format(NN_model_path))
    print('Working directory       : {0}'.format(work_dir_path))
    print('__________________________________________________________________________________________________')
    print('Keras data format       : {0}'.format(K.image_data_format()))
    print('Optimizer               : {0}'.format(optimizer.__class__.__name__))
    print('Loss                    : {0}'.format(loss.__name__))
    print('Metrics                 : {0}'.format(model.metrics_names[1:]))      # model.metrics_names[0] = 'loss'
    print('Monitor for best        : {0}'.format(LR_params['monitor_for_best']))
    print('Patience for early stop : {0}'.format(patience_for_stop))
    print('Custom layers           : {0}'.format(list(custom_layers.keys()) ))
    print('Batch size              : {0}'.format(NN_batch_size))
    print('Epochs                  : {0} - {1}'.format(init_epoch+1, number_of_epochs))
    print('Learning rate formula   : {0}'.format(LR_params['formula']))
    print('Learning rate graph     : {0}'.format(LR_params['graph']))
    print('LR step                 : {0}'.format(LR_params['step']))
    print('LR limit                : {0}'.format(LR_params['limit']))
    print('Patience for LR step    : {0}'.format(LR_params['patience']))
    print('__________________________________________________________________________________________________')
    if len(NN_info) > 0:
        for info in NN_info: print(info)
        print('__________________________________________________________________________________________________')


    # Checkpoint
    # key = input('Continue? [y/n] : ')
    # if key != 'y' and key != 'Y':
    #     print('Exit...')
    #     sys.exit()


    # Define callbacks
    autoLR = AutoLRManager(param=LR_params, early_stop=patience_for_stop)

    def LRScheduler(epoch, lr):
        import math
        raw_lr = lr

        if LR_params['formula'][0] == 'poly':
            # See https://arxiv.org/pdf/1506.04579.pdf
            print('Learning rate by poly: base_lr = {0}, power = 0.9'.format(LR_params['formula'][1]))
            raw_lr = LR_params['formula'][1] * math.pow(1 - epoch / number_of_epochs, 0.9)

        # elif LR_params['formula'][0] == 'XXX':
        #     print('Learning rate by XXX: ...
        #     raw_lr = LR_params['formula'][1] ...

        elif LR_params['graph'] is not None:
            def LR_at_epoch(epoch, pt1, pt2): return (pt2[1] - pt1[1]) / (pt2[0] - pt1[0]) * (epoch - pt1[0]) + pt1[1]
            print('Learning rate by graph [epoch, LR] : {0}'.format(LR_params['graph']))
            for i in range(len(LR_params['graph'])-1):
                if LR_params['graph'][i][0] <= epoch and epoch < LR_params['graph'][i+1][0]:
                    raw_lr = LR_at_epoch(epoch, LR_params['graph'][i], LR_params['graph'][i+1])
                    break

        decay = autoLR.get_LR_decay()
        new_LR = decay * raw_lr
        print('LR = {0} (raw LR = {1}, decay = {2})'.format(new_LR, raw_lr, decay))
        return new_LR

    print('\n- Defining callbacks...')
    checkpointer = ModelCheckpoint(tmp_model_path, monitor=LR_params['monitor_for_best'][0], verbose=1, save_best_only=True, mode=LR_params['monitor_for_best'][1])
    earlyStopper = EarlyStopping(monitor=LR_params['monitor_for_best'][0], min_delta=0, patience=patience_for_stop, verbose=1, mode=LR_params['monitor_for_best'][1])
    scheduleLR = LearningRateScheduler(LRScheduler, verbose=0)
    csvlogger = CSVLogger(os.path.join(work_dir_path,'training_log.csv'), separator=',', append=False)
    # tensorboard = TensorBoard(log_dir=work_dir_path, histogram_freq=0, write_graph=True, write_images=True)


    # Image data generator
    '''
        For TensorFlow 2.0
        Model.fit_generator IS DEPRECATED.
        To use Model.fit, generator classes, ImageDataGenerator_XXX(), were updated as subclasses of keras.utils.Sequence.

        See:
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit_generator
    '''
    from utils.Image_data_generator import ImageDataGenerator_CSV_with_Header, ImageDataGenerator_h5_Dataset
    print('\n- Loading images for training...')
    ext = os.path.splitext(sys.argv[2])[1]
    if   ext == '.csv' :  training_images = ImageDataGenerator_CSV_with_Header('Train data from CSV', sys.argv[2], batch_size=NN_batch_size, rescale=1.0/225.0, shuffle=True)
    elif ext == '.h5'  :  training_images = ImageDataGenerator_h5_Dataset('image_training', sys.argv[2], batch_size=NN_batch_size, rescale=1.0/225.0)
    else               :  sys.exit()
    print('\n- Loading images for validation...')
    ext = os.path.splitext(sys.argv[3])[1]
    if   ext == '.csv' :  validation_images = ImageDataGenerator_CSV_with_Header('Validation data from CSV', sys.argv[3], batch_size=NN_batch_size, rescale=1.0/225.0, shuffle=True)
    elif ext == '.h5'  :  validation_images = ImageDataGenerator_h5_Dataset('image_validation', sys.argv[3], batch_size=NN_batch_size, rescale=1.0/225.0)
    else               :  sys.exit()


    # Save network figure and parameters
    plot_model(model, to_file=os.path.join(work_dir_path,'model_figure.png'), show_shapes=True, show_layer_names=False)
    with open(os.path.join(work_dir_path,'training_parameters.txt'), mode='w') as path_file:
        path_file.write('Date                    : {0}\n'.format(startdate))
        path_file.write('TensorFlow version      : {0}\n'.format(tf.version.VERSION))
        path_file.write('Keras version           : {0}\n'.format(tf.keras.__version__))
        path_file.write('OS-version              : {0}\n'.format(platform.platform()))
        path_file.write('Processor               : {0}\n\n'.format(platform.processor()))
        path_file.write('Training mode           : {0}\n'.format(training_mode))
        path_file.write('Model name              : {0}\n'.format(NN_model_name))
        path_file.write('Model description       : {0}\n'.format(NN_model_descript))
        path_file.write('Number of classes       : {0}\n'.format(NN_num_classes))
        path_file.write('Loaded model path       : {0}\n'.format(NN_model_path))
        path_file.write('Working directory       : {0}\n\n'.format(work_dir_path))
        path_file.write('Training images         : {0} sets in {1}\n'.format(training_images.length(), sys.argv[2]))
        path_file.write('Validation images       : {0} sets in {1}\n\n'.format(validation_images.length(), sys.argv[3]))
        path_file.write('Keras data format       : {0}\n'.format(K.image_data_format()))
        path_file.write('Optimizer               : {0}\n'.format(optimizer.__class__.__name__))
        path_file.write('Loss                    : {0}\n'.format(loss.__name__))
        path_file.write('Metrics                 : {0}\n'.format(model.metrics_names[1:]))
        path_file.write('Monitor for best        : {0}\n'.format(LR_params['monitor_for_best']))
        path_file.write('Patience for early stop : {0}\n'.format(patience_for_stop))
        path_file.write('Custom layers           : {0}\n'.format(list(custom_layers.keys()) ))
        path_file.write('Batch size              : {0}\n'.format(NN_batch_size))
        path_file.write('Epochs                  : {0} - {1}\n'.format(init_epoch+1, number_of_epochs))
        path_file.write('Learning rate formula   : {0}\n'.format(LR_params['formula']))
        path_file.write('Learning rate graph     : {0}\n'.format(LR_params['graph']))
        path_file.write('LR step                 : {0}\n'.format(LR_params['step']))
        path_file.write('LR limit                : {0}\n'.format(LR_params['limit']))
        path_file.write('Patience for LR step    : {0}\n\n'.format(LR_params['patience']))
        if len(NN_info) > 0:
            for info in NN_info: path_file.write('{}\n'.format(info))
        path_file.write('\n')
        model.summary(print_fn=lambda x: path_file.write(x + '\n'))


    # Train the model
    '''
        For TensorFlow 2.0
        fit_generator -> fit

        Warning: Model.fit_generator IS DEPRECATED. It will be removed in a future version.
        Instructions for updating: Please use Model.fit, which supports generators.

        See:
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit
        https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit_generator
    '''
    print('\n- Starting model lerning...')
    results = model.fit_generator(training_images.flow(),
        steps_per_epoch         = training_images.length() // NN_batch_size,
        epochs                  = number_of_epochs,
        verbose                 = 1,
        callbacks               = [checkpointer, earlyStopper, autoLR, scheduleLR, csvlogger],
        validation_data         = validation_images.flow(),
        validation_steps        = validation_images.length() // NN_batch_size,
        max_queue_size          = 2,
        workers                 = 1,
        use_multiprocessing     = False,
        shuffle                 = False,
        initial_epoch           = init_epoch )

    ##### For TensorFlow v2.0 #####
    # results = model.fit(
    #     x                       = training_images,      # keras.utils.Sequence
    #     epochs                  = number_of_epochs,
    #     verbose                 = 1,
    #     # callbacks               = [checkpointer, earlyStopper, autoLR, scheduleLR, csvlogger, tensorboard],
    #     callbacks               = [checkpointer, earlyStopper, autoLR, scheduleLR, csvlogger],
    # 	validation_data         = validation_images.getdata(),  # tuple of Numpy arrays
    #     shuffle                 = False,
    #     initial_epoch           = init_epoch,
    #     validation_freq         = 1,
    #     max_queue_size          = 10,
    #     workers                 = 10,
    #     use_multiprocessing     = False )


    # Show results
    print('\n- Saving training graph...')
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

    ax1.plot(xlen, his_loss, marker='.', color='salmon', label=loss.__name__)
    ax1.plot(xlen, his_valloss, marker='.', color='red', label='val_'+loss.__name__)
    ax2.plot(xlen, his_met1, marker='.', color='deepskyblue', label=model.metrics_names[1])
    ax2.plot(xlen, his_valmet1, marker='.', color='blue', label='val_'+model.metrics_names[1])
    ax2.plot(xlen, his_met2, marker='.', color='limegreen', label=model.metrics_names[2])
    ax2.plot(xlen, his_valmet2, marker='.', color='green', label='val_'+model.metrics_names[2])

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel(loss.__name__)
    ax1.set_yscale("log")
    ax1.set_ylim([0.001, 1.0])
    ax2.set_ylabel('Metrics')
    ax2.set_yscale("log")
    ax2.set_ylim([0.7, 1.0])

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1+h2, l1+l2, loc='lower center')

    plt.savefig(os.path.join(work_dir_path,'training_graph.png'))
    # plt.show()


    # Save the trained model
    print('\n- Saving trained model...')
    if LR_params['monitor_for_best'][1] == 'max': score = max(results.history[LR_params['monitor_for_best'][0]])
    else:                                         score = min(results.history[LR_params['monitor_for_best'][0]])
    save_name = 'model{0}, {1}={2:.4f}, {3} by {4}.h5'.format(datestr, LR_params['monitor_for_best'][0], score, training_mode, NN_model_name)
    save_path = os.path.join(work_dir_path, save_name)
    model.save(save_path)
    time.sleep(20)
    print('Final model path: ' + save_path)

    if os.path.exists(tmp_model_path):
        os.remove(tmp_model_path)
        print('Temp model removed: ' + tmp_model_path)


    print('\n==================================================================================================')
    print('Computation time        : {0}'.format(timedelta(seconds=time.time()-starttime)))
    print('From the date           : {0}\n'.format(startdate))
    print('==================================================================================================')




# Main
if __name__ == '__main__': Train()



