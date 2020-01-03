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
    print('  argv[1] : Path to a directory to save generated connverted CoreML model in it')
    print('  argv[2] : Model with custom Keras metrics? (y/n)')
    print('  argv[3] : Path to a Keras model file (.h5) to be converted')
    print('  argv[4] : Path to a relevant Keras model code (.py)')
    sys.exit()




'''
    Definition of custom loss and metrics used during training with Keras
    The following list was copied from training management code, Train.py
    And the Validation_func.py must be the same as that used during training by Train.py
'''
# Select loss function used in the Keras model to be converted
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
#custom_metrics = {'mean_iou': mean_iou, 'mean_rou': mean_rou, 'mean_iou_rou': mean_iou_rou, 'dice_coef': dice_coef}




'''
    Definition of custom layers that might be used in neural network models
    Those layers are supposed to be implemented in A.I.Segmentation for inference
    Do not change the classNmae_XXX because it is used to identify its relevant class in Keras code
'''
from coremltools.proto import NeuralNetwork_pb2

# GELU converter
className_GELU = 'GELU'
def convert_GELU(keras_layer):
    coreml_layer = NeuralNetwork_pb2.CustomLayerParams()
    coreml_layer.className = className_GELU
    coreml_layer.description = 'Custom activation layer: ' + className_GELU
    return coreml_layer

# Swish converter
className_Swish = 'Swish'
def convert_Swish(keras_layer):
    coreml_layer = NeuralNetwork_pb2.CustomLayerParams()
    coreml_layer.className = className_Swish
    coreml_layer.description = 'Custom activation layer: ' + className_Swish
    return coreml_layer

# Dictionary of Keras custom layers implemented in A.I.Segmentation
supported_layers_in_AIAS = { className_GELU: convert_GELU,
                            className_Swish: convert_Swish }




'''
    Main converter

    print('  argv[1] : Path to a directory to save generated connverted CoreML model in it')
    print('  argv[2] : Model with custom Keras metrics? (y/n)')
    print('  argv[3] : Path to a Keras model file (.h5) to be converted')
    print('  argv[4] : Path to a relevant Keras model code (.py)')
'''
import os
import warnings
#warnings.filterwarnings('ignore')
from coremltools.converters.keras._keras_converter import convertToSpec
from coremltools.models import MLModel, _MLMODEL_FULL_PRECISION, _MLMODEL_HALF_PRECISION
import numpy as np


# Convert Keras model to CoreML model
def convert_keras_to_mlmodel(keras_model_path, coreml_model_path, w_custom_objects):

    NN_name = os.path.splitext(os.path.basename(sys.argv[4]))[0]
    print('----------------------------------------------------------')
    print('NN model file:')
    print(sys.argv[4])
    print('NN model name:')
    print(NN_name)

    print('----------------------------------------------------------')
    print('Keras model file:')
    print(keras_model_path)
    print('CoreML model file:')
    print(coreml_model_path)

    print('----------------------------------------------------------')
    print('Keras custom layers implemented in AIAS for this code:')
    for k in supported_layers_in_AIAS: print(k)


    # Import neural network code
    import importlib.machinery as imm
    NN = imm.SourceFileLoader(NN_name, sys.argv[4]).load_module()


    # Load custom layers if implemented in each Keras model
    # Take care the imported NN may not have Custom_Layers() def, so try and catch except
    try:    keras_custom_layers = NN.Custom_Layers()
    except: keras_custom_layers = {}


    # Construct custom layers and conversion functions
    custom_conversion_functions = {}
    custom_layers = {}
    print('----------------------------------------------------------')

    if keras_custom_layers is not None:
        print('Custom layers in this Keras model:')
        for keras_layer_key in keras_custom_layers:
            if keras_layer_key in supported_layers_in_AIAS:
                print(keras_layer_key + ' - available')
                custom_layers[keras_layer_key] = keras_custom_layers[keras_layer_key]
                custom_conversion_functions[keras_layer_key] = supported_layers_in_AIAS[keras_layer_key]
            else:
                print(keras_layer_key + ' - unavailable')

        print('Matched layers and conversion functions for coremltools:')
        print(custom_layers)
        print(custom_conversion_functions)

    else:
        print('Custom layers not found in this Keras model.')


    # Construct custom objects (= loss + metrics + layers)
    if w_custom_objects == 'y':
        custom_objects = {LOSS.__name__: LOSS, 'mean_iou': mean_iou, 'mean_rou': mean_rou, 'mean_iou_rou': mean_iou_rou, 'dice_coef': dice_coef}
    elif w_custom_objects == 'n':
        custom_objects = {}
    else:
        print('Use (y) or (n) for argv[2] : Model with custom Keras metrics?')
        sys.exit()

    print('----------------------------------------------------------')
    print('Custom objects for coremltools:')
    print(dict(**custom_objects, **custom_layers))
    print('----------------------------------------------------------')


    # Convert
    # Do not change the input_names/output_names because they are used to identify input/output layers in Keras code
    spec = convertToSpec( keras_model_path,
                          input_names                   = 'input',
                          output_names                  = 'output',
                          add_custom_layers             = True,
                          custom_conversion_functions   = custom_conversion_functions,
                          custom_objects                = dict(**custom_objects, **custom_layers),
                          respect_trainable             = False )       # should be True???
    model = MLModel(spec)


    # Set descriptions
    model.author = 'Takashi Shirakawa'
    model.license = '(C) 2019-2020, Takashi Shirakawa. All right reserved.'
    model.short_description = 'CoreML model for A.I.Segmentation'
    model.input_description['input'] = 'Input grayscale image must be 8-bit depth (0-255 tones) per pixel with the size of 200x200 pixels.'
    model.output_description['output'] = 'Predicted images (segmentation results) will be saved in the same format.'


    # Save mlmodel
    model.save(coreml_model_path)


    # Show results
    print('----------------------------------------------------------')
    print('Model descriptions:')
    print(spec.description)

    print('Custom layers:')
    for i, layer in enumerate(spec.neuralNetwork.layers):
        if layer.HasField('custom'):
            print('Layer %d = %s : class name = %s' % (i+1, layer.name, layer.custom.className))
#        else:
#            print('Layer %d = %s' % (i, layer.name))

    print('Done.')




# Main
if __name__ == '__main__':
    convert_keras_to_mlmodel(sys.argv[3], os.path.join(sys.argv[1], os.path.splitext(os.path.basename(sys.argv[3]))[0] + '.mlmodel'), sys.argv[2])



