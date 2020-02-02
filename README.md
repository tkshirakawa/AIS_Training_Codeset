# AIS Training Codeset
Source codes to create macOS CoreML models for A.I.Segmentation plugin.<br>
This codeset contains
1. Preprocessing codes for image augmentation, storing data in HDF5, and so on.
1. A training management code. Variable learning rate is available.
1. Codes for custom loss, metrics and layers for advanced training.
1. Neural network models of Keras+TensorFlow: CV-net SYNAPSE, CV-net2, CV-net, U-net and DeepLab v3+ *1.
1. A converter code to convert the trained Keras model to a CoreML model for macOS *2.

*1 Original sources of neural network models<br>
U-net : chuckyee/cardiac-segmentation, https://github.com/chuckyee/cardiac-segmentation<br>
DeepLab v3+ : bonlime/keras-deeplab-v3-plus, https://github.com/bonlime/keras-deeplab-v3-plus<br>

*2 You need to install coremltools from Apple to use the converter<br>
coremltools : Apple, https://github.com/apple/coremltools

<img width="1223" alt="ss_v20" src="https://user-images.githubusercontent.com/52600509/71913629-3705e500-31bb-11ea-9226-3885f33f82c3.png">
