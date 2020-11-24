# AIS Training Codeset
This codeset is a set of Python codes to train neural network models of Keras with your original dataset [1] for semantic segmentation. You can use your trained models in Python command lines as usual. Moreover, you can convert the models into macOS Core ML models which are loadable for A.I.Segmentation [2], a macOS GUI plugin for semantic segmentation of medical images in DICOM data.

> [1] A dataset is pairs of images and grandtruth masks (**Fig.1**). Images trainable in this system are squre and 8-bit grayscale w/o alpha channel. **See [How to Use AIS Training Codeset](https://github.com/tkshirakawa/AIS_Training_Codeset/blob/master/How%20to%20Use%20AIS%20Training%20Codeset.pdf) for more details.**  
> [2] A.I.Segmentation is a simple plugin for OsiriX.  
> [A.I.Segmentation](https://compositecreatures.jimdofree.com/a-i-segmentation/), an OsiriX plugin that performs semantic segmentation based on Apple's Core ML technology.  
> [OsiriX](https://www.osirix-viewer.com), the most advanced DICOM viewer for macOS than ever before.  
> [Core ML](https://developer.apple.com/machine-learning/core-ml/), Apple's API for neural network computation.  
<br>

<img width="170" alt="dataset" src="https://user-images.githubusercontent.com/52600509/92623102-ccd0b180-f300-11ea-83e8-456f8acb50a2.png">

**Figure 1.** A dataset sample of a cardiac CT image and segmentation mask of the heart.

<br>
<br>

## Description
Before you use this codeset and system, you need:
- **Computer with GPU or Neural engine** - CPU-only computers are not good for a large training dataset.
- **Windows or Linux** - TensorFlow is optimized for those OSs. *Mac with Apple Silicon may have good performance with TensorFlow.
- **Python 3.7.7 / Keras 2.2.4 / TensorFlow 1.15.0** - TF 2.* is available but you may have compatibility troubles when converting the model into macOS Core ML format.
- **Your dataset** - the most important thing is giving accurate and precise segmentation masks.

This codeset contains:
- Preprocessing codes for image augmentation, list generation of images, HDF5 archiving, etc.
- A training management code with a learning rate controller and metrics monitor.
- Custom loss, metrics and layers for advanced training [1].
- Neural network models of Keras: CV-net SYNAPSE, U-net and DeepLab v3+ [2].
- A converter code to convert a trained Keras model to a Core ML model for macOS [3].

> [1] Special losses and metrics implemented in this codeset  
> Focal CE loss: Tsung-Yi Lin, et al., Focal Loss for Dense Object Detection, [arXivk, 2018(v2)](https://arxiv.org/abs/1708.02002v2)  
> Constrained focal CE loss: Y. Zhao, et al., Constrained-Focal-Loss Based Deep Learning for Segmentation of Spores, [IEEE Access vol.7, 2019](https://ieeexplore.ieee.org/document/8896836)  
> Hausdorff distance loss: Karimi D, et al., Reducing the Hausdorff Distance in Medical Image Segmentation With Convolutional Neural Networks, [IEEE Trans Med Imaging, 2020](https://ieeexplore.ieee.org/document/8767031)  
> [2] Neural network models  
> **CV-net SYNAPSE: my original neural network**  
> U-net: implementation by chuckyee - [chuckyee/cardiac-segmentation in GitHub](https://github.com/chuckyee/cardiac-segmentation)  
> DeepLab v3+: implementation by bonlime - [bonlime/keras-deeplab-v3-plus in GitHub](https://github.com/bonlime/keras-deeplab-v3-plus)  
> [3] You need to install coremltools by Apple to use the converter - [coremltools by Apple](https://github.com/apple/coremltools)  

<br>

<img width="650" alt="training_flow" src="https://user-images.githubusercontent.com/52600509/92629460-4bc9e800-f309-11ea-8250-17afd7ccd838.png">
<br>

## Training Flow
1. Prepare your dataset: public data from web, personal data in your PC, and/or any images.
1. Locate the dataset in directories with prearranged names. Follow the rules for dataset.
1. Inflate the dataset by data augumentation technique.
1. Make a CSV list of the augumented dataset. Separate the list into training and validation data.
1. Convert those datasets in CSV lists into HDF5 files.
1. Training by Keras+TensorFlow.
1. Convert the trained Keras model to a Core ML model to use it in A.I.Segmentation for direct segmentation in OsiriX DICOM viewer for macOS (**Fig.2**).

**Please refer to a PDF document: [How to Use AIS Training Codeset](https://github.com/tkshirakawa/AIS_Training_Codeset/blob/master/How%20to%20Use%20AIS%20Training%20Codeset.pdf) for more details.**
<br>
<br>
<br>

<img width="800" alt="ss_v20" src="https://user-images.githubusercontent.com/52600509/71913629-3705e500-31bb-11ea-9226-3885f33f82c3.png">

**Figure 2.** A.I.Segmentation plugin draws segmentation ROIs on DICOM images based on a selected neural network model of Core ML.

<br>
<br>

## License
Copyright (c) 2019-2020, Takashi Shirakawa. All rights reserved.

Released under the MIT license.  
https://opensource.org/licenses/mit-license.php

