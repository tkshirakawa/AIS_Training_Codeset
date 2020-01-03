# AIS Training Codeset
Source codes to create macOS CoreML models for A.I.Segmentation plugin.
This codeset contains
1) Preprocessing codes for image augmentation and storing in HDF5
2) A training management code with custom loss, metrics, layers and variable learning rate
3) Neural network models of Keras+TensorFlow: CV-net, U-net and DeepLab v3+
4) A converter code to convert the trained Keras model to a CoreML model for macOS
