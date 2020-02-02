# AIS Training Codeset
Source codes to create macOS CoreML models for A.I.Segmentation plugin.

## Description
This codeset contains
- Preprocessing codes for image augmentation, storing data in HDF5, and so on.
- A training management code. Variable learning rate is available.
- Codes for custom loss, metrics and layers for advanced training.
- Neural network models of Keras+TensorFlow: CV-net SYNAPSE, CV-net2, CV-net, U-net and DeepLab v3+ *1.
- A converter code to convert the trained Keras model to a CoreML model for macOS *2.

*1 Original sources of neural network models<br>
U-net : chuckyee/cardiac-segmentation, https://github.com/chuckyee/cardiac-segmentation<br>
DeepLab v3+ : bonlime/keras-deeplab-v3-plus, https://github.com/bonlime/keras-deeplab-v3-plus<br>

*2 You need to install coremltools from Apple to use the converter<br>
coremltools : Apple, https://github.com/apple/coremltools<br>

---
## Training Flow
Please refer to 'How to Use AIS Training Codeset' https://github.com/tkshirakawa/AIS_Training_Codeset/blob/master/How%20to%20Use%20AIS%20Training%20Codeset.pdf

1. Prepare your dataset: public data from web, personal data in your PC, and/or any images.
1. Locate the dataset in directories with prearranged names. Follow the rules for dataset.
1. Inflate the dataset by data augumentation technique.
1. Make CSV lists of the dataset. Save separately for training and validation.
1. Convert the dataset in the CSV lists into HDF5.
1. Training by Keras+TensorFlow.
1. Convert the trained Keras model to a Core ML model for AIS in macOS.

<img width="1223" alt="ss_v20" src="https://user-images.githubusercontent.com/52600509/71913629-3705e500-31bb-11ea-9226-3885f33f82c3.png">
