'''
    Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.
    e-mail: tkshirakawa@gmail.com

    Released under the BSD 3-Clause License
'''




import sys
import os
import warnings
warnings.filterwarnings('ignore')

if sys.argv[1] == '-h':
    print('### Help for argv ###')
    print('  argv[1] : Path to a directory to save a connverted CoreML model in it.')
    print('  argv[2] : Path to a trained Keras model file (.h5) to be converted.')
    print('  argv[3] : Path to a Keras model code (.py) used for the training.')
    print('  argv[4] : Path to Train.py used for the training.')
    print('  # Custom layers that may be used in the model will be loaded automatically from the model.')
    sys.exit()




'''
    Definition of custom layers that might be used in neural network models
    Those layers are supposed to be implemented in A.I.Segmentation for inference
    Do not change the classNmae_XXX because it is used to identify its relevant class in Keras code
'''
import keras.backend as K
from coremltools.proto import NeuralNetwork_pb2


call_count = 1

# SynapticTransmissionRegulator converter
className_SynapticTransmissionRegulator = 'SynapticTransmissionRegulator'
def convert_SynapticTransmissionRegulator(keras_layer):
    coreml_layer = NeuralNetwork_pb2.CustomLayerParams()
    coreml_layer.className = className_SynapticTransmissionRegulator
    coreml_layer.description = 'Custom Synaptic Transmission Regulator (STR) layer: ' + className_SynapticTransmissionRegulator
    
    weightList = keras_layer.get_weights()
    p_weight = weightList[0]     # numpy array
    p_bias   = weightList[1]     # numpy array
    weight = coreml_layer.weights.add()
    weight.floatValue.extend(map(float, p_weight.flatten()))
    bias = coreml_layer.weights.add()
    bias.floatValue.extend(map(float, p_bias.flatten()))

    return coreml_layer

'''
### Sample code of PReLU (Parametric Rectified Linear Unit) from Keras ###

- Definition of the layer found in 'advanced_activations.py' in Keras.
class PReLU(Layer):
    ...
    def build(self, input_shape):
        param_shape = list(input_shape[1:])
        self.alpha = self.add_weight(shape=param_shape, name='alpha', ...


- Definition of its converter found in '_layers.py' in coremltools.
def convert_activation(builder, layer, input_names, output_names, keras_layer):
    ...
    params = keras.backend.eval(keras_layer.weights[0])

    builder.add_activation(
        name = layer,
        non_linearity = non_linearity,
        input_name = input_name,
        output_name = output_name,
        params = params)


def add_activation(self, name, non_linearity, input_name, output_name, params=None):
    ...
    params: list of float or numpy.array. Parameters for the activation, depending on non_linearity.

    - When non_linearity is 'PRELU', param is a list of 1 numpy array [alpha]. The shape of
      alpha is (C,), where C is either the number of input channels or 1. When C = 1, same alpha is applied to all channels.
    
    ...
    elif non_linearity == 'PRELU':
        # PReLU must provide an np array in params[0]
        spec_layer_params.PReLU.alpha.floatValue.extend(map(float, params.flatten()))
'''



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


# Square converter
className_Square = 'Square'
def convert_Square(keras_layer):
    coreml_layer = NeuralNetwork_pb2.CustomLayerParams()
    coreml_layer.className = className_Square
    coreml_layer.description = 'Custom layer: ' + className_Square
    return coreml_layer


# SQRT converter
className_SQRT = 'SQRT'
def convert_SQRT(keras_layer):
    coreml_layer = NeuralNetwork_pb2.CustomLayerParams()
    coreml_layer.className = className_SQRT
    coreml_layer.description = 'Custom layer: ' + className_SQRT
    return coreml_layer


# Dictionary of Keras custom layers implemented in A.I.Segmentation
conversion_func_in_AIAS = { className_SynapticTransmissionRegulator : convert_SynapticTransmissionRegulator,
                                                     className_GELU : convert_GELU,
                                                    className_Swish : convert_Swish,
                                                   className_Square : convert_Square,
                                                     className_SQRT : convert_SQRT }




'''
    Main converter: convert Keras model to CoreML model
'''
def convert_keras_to_mlmodel(keras_model_path, coreml_model_path):

    import importlib.machinery as imm
    from coremltools.converters.keras._keras_converter import convertToSpec
    from coremltools.models import MLModel, _MLMODEL_FULL_PRECISION, _MLMODEL_HALF_PRECISION
#    from coremltools.models.utils import convert_double_to_float_multiarray_type

    from keras.models import load_model
    from kerassurgeon.operations import delete_layer


    sys.path.append(os.path.dirname(sys.argv[4]))


    # Import neural network code
    NN_file_name = os.path.splitext(os.path.basename(sys.argv[3]))[0]
    NN = imm.SourceFileLoader(NN_file_name, sys.argv[3]).load_module()
    
    try:    NN_model_name = NN.Model_Name()
    except: NN_model_name = NN_file_name
    
    try:    NN_model_description = NN.Model_Description()
    except: NN_model_description = None


    # Load custom layers if implemented in each Keras model
    # Take care the imported NN may not have Custom_Layers() def, so try and catch except.
    # The type is a dictionary. The keys are supposed to be same as the corresponding values (=defs).
    try:    NN_custom_layers = NN.Custom_Layers()
    except: NN_custom_layers = {}


    # Import Train.py to get custom loss and metrics
    Train_name = os.path.splitext(os.path.basename(sys.argv[4]))[0]
    Train_py = imm.SourceFileLoader(Train_name, sys.argv[4]).load_module()

    custom_loss = Train_py.get_loss()
    custom_metrics = Train_py.get_metrics()

    kpt, kex = os.path.splitext(keras_model_path)
    keras_model_path_temp = kpt + '_temp' + kex

    print('----------------------------------------------------------')
    print('NN model file path: {}'.format(sys.argv[3]))
    print('NN model name: {}'.format(NN_model_name))
    print('NN model description: {}'.format(NN_model_description))
    print('NN custom layers:')
    print(NN_custom_layers)
    print('Training file path and loss/metrics used:')
    print(sys.argv[4])
    print(custom_loss)
    print(custom_metrics)

    print('----------------------------------------------------------')
    print('Keras model file: {}'.format(keras_model_path))
    print('Keras model file temp: {}'.format(keras_model_path_temp))
    print('CoreML model file: {}'.format(coreml_model_path))

    print('----------------------------------------------------------')
    print('Keras custom layers implemented in AIAS for this code:')
    for k in conversion_func_in_AIAS: print(k)


    # Deleting Dropout layers from the Keras model to be converted
    # Because the layers will cause unknown conversion failures in coremltools
    keras_model = load_model(keras_model_path, custom_objects=dict(**custom_loss, **custom_metrics, **NN_custom_layers), compile=False)

    print('----------------------------------------------------------')
    keras_model.summary()

    idp = 1
    del_prefix = 'dropout'
    while True:
        try:    layer = keras_model.get_layer('{}_{}'.format(del_prefix, idp))
        except: break
        print('Deleting layer: {}_{}'.format(del_prefix, idp))
        keras_model = delete_layer(model=keras_model, layer=layer, copy=False)
        idp += 1

    keras_model.summary()
    print('Saving temporary Keras model: {}'.format(keras_model_path_temp))
    keras_model.save(keras_model_path_temp)


    # Construct custom layers and conversion functions
    custom_layers = {}
    custom_conversion_func = {}
    print('----------------------------------------------------------')

    if NN_custom_layers is not None:
        print('Custom layers in this Keras model:')
        for keras_layer_key in NN_custom_layers:
            if keras_layer_key in conversion_func_in_AIAS:
                print(keras_layer_key + ' - available')
                custom_layers[keras_layer_key]          = NN_custom_layers[keras_layer_key]
                custom_conversion_func[keras_layer_key] = conversion_func_in_AIAS[keras_layer_key]
            else:
                print(keras_layer_key + ' - unavailable')

        print('Matched layers and conversion functions for coremltools:')
        print(custom_layers)
        print(custom_conversion_func)

    else:
        print('Custom layers not found in this Keras model.')


    custom_objects = dict(**custom_loss, **custom_metrics, **custom_layers)

    print('----------------------------------------------------------')
    print('Custom objects passed into coremltools converter:')
    print(custom_objects)
    print('----------------------------------------------------------')


    # Convert
    # Do not change the input_names/output_names because they are used to identify input/output layers in Keras code
    spec = convertToSpec( keras_model_path_temp,
                          input_names                 = 'input',
                          output_names                = 'output',
                          add_custom_layers           = True,
                          custom_conversion_functions = custom_conversion_func,
                          custom_objects              = custom_objects,
                          respect_trainable           = False )       # should be True???
    model = MLModel(spec)


    # Set descriptions
    model.author = 'Takashi Shirakawa'
    model.license = '(C) 2019-2020, Takashi Shirakawa. All right reserved.'
    model.short_description = NN_model_name + ' for A.I.Segmentation'
    model.input_description['input'] = 'Input grayscale image must be 8-bit depth (0-255 tones) per pixel with the size of 200x200 pixels.'
    model.output_description['output'] = 'Predicted images (segmentation results) will be saved in the same format.'


    # Save mlmodel
    model.save(coreml_model_path)
    
#    spec_f = model.get_spec()
#    convert_double_to_float_multiarray_type(spec_f)
#    model_f = MLModel(spec_f)
#    model_f.save(os.path.splitext(coreml_model_path)[0] + ', float_multiarray.mlmodel')


    # Show results
    spec = model.get_spec()
    print('----------------------------------------------------------')
    print('Model descriptions:')
    print(spec.description)
#    print('Model descriptions (float multiarray type):')
#    print(spec_f.description)

    print('Custom layers:')
    for i, layer in enumerate(spec.neuralNetwork.layers):
        if layer.HasField('custom'):
            print('Layer %d = %s : class name = %s' % (i+1, layer.name, layer.custom.className))
#        else:
#            print('Layer %d = %s' % (i, layer.name))

    print('Done.')




# Main
if __name__ == '__main__':
    convert_keras_to_mlmodel(sys.argv[2], os.path.join(sys.argv[1], os.path.splitext(os.path.basename(sys.argv[2]))[0] + '.mlmodel'))



