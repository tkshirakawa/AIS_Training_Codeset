# AIS Training Codeset
Source codes to create macOS CoreML models for A.I.Segmentation plugin.
This codeset contains
1) Preprocessing codes for image augmentation and storing in HDF5
2) A training management code with custom loss, metrics, layers and variable learning rate
3) Neural network models of Keras+TensorFlow: CV-net, U-net and DeepLab v3+ *
4) A converter code to convert the trained Keras model to a CoreML model for macOS

*Original sources of neural network models<br>
U-net : chuckyee/cardiac-segmentation, https://github.com/chuckyee/cardiac-segmentation<br>
DeepLab v3+ : bonlime/keras-deeplab-v3-plus, https://github.com/bonlime/keras-deeplab-v3-plus<br>

<img width="416" alt="panel_v20" src="https://user-images.githubusercontent.com/52600509/71713642-e9514b80-2e4d-11ea-9f91-8ece251c9eff.png">
