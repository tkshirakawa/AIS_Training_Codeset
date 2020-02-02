# AIS Training Codeset
Python code to train neural network models with your original dataset for semantic segmentation. This codeset also includes a converter to create macOS Core ML models for A.I.Segmentation plugin for OsiriX.<br>
<br>
- Library for training: **Keras with TensorFlow backend**
- Library for prediction: **Core ML in macOS**
- Data: **8-bit grayscale image w/o alpha channel, 200x200 pixel size**<br>
        (That means each pixel has just one channel of unsigned integer value 0-255)

Please refer to PDF: [How to Use AIS Training Codeset](https://github.com/tkshirakawa/AIS_Training_Codeset/blob/master/How%20to%20Use%20AIS%20Training%20Codeset.pdf) for more details.<br>
<br>

## Description
This codeset contains:
- Preprocessing code for image augmentation, etc.
- A training management code. Variable learning rate is available.
- Codes for custom loss, metrics and layers for advanced training.
- Neural network models of Keras+TensorFlow: CV-net SYNAPSE, CV-net2, CV-net, U-net and DeepLab v3+ *1.
- A converter code to convert the trained Keras model to a CoreML model for macOS *2.

*1 Original sources of neural network models are<br>
U-net : by chuckyee, see [chuckyee/cardiac-segmentation](https://github.com/chuckyee/cardiac-segmentation)<br>
DeepLab v3+ : by bonlime, see [bonlime/keras-deeplab-v3-plus](https://github.com/bonlime/keras-deeplab-v3-plus)<br>

*2 You need to install coremltools from Apple to use the converter. See [coremltools by Apple](https://github.com/apple/coremltools).<br>
<br>

## Training Flow
1. Prepare your dataset: public data from web, personal data in your PC, and/or any images.
1. Locate the dataset in directories with prearranged names. Follow the rules for dataset.
1. Inflate the dataset by data augumentation technique.
1. Make a CSV list of the augumented dataset. Separate the list into training and validation data.
1. Convert those datasets in CSV lists into HDF5 files.
1. Training by Keras+TensorFlow.
1. Convert the trained Keras model to a Core ML model for AIS in macOS.

More: [How to Use AIS Training Codeset](https://github.com/tkshirakawa/AIS_Training_Codeset/blob/master/How%20to%20Use%20AIS%20Training%20Codeset.pdf)<br>
<br>
<br>

<img width="1223" alt="ss_v20" src="https://user-images.githubusercontent.com/52600509/71913629-3705e500-31bb-11ea-9226-3885f33f82c3.png">
