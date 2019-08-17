# SCSNet

The test code for my CVPR2019 paper "Scalable Convolutional Neural Network for Image Compressed Sensing" will be released here. Please look forward to it.

%-----------------2019.08.17---------------------
The codes provide two network structures, of which one learns the basic reconstruction layer and all enhancement reconstruction layers simutaneously, and the other one learns each enhancement reconstruction layer independantly. The former one requires large CPU and GPU memory, so the later training mode is recommended. Note that when each enhancement layer is trained independetly, the learning rate of the parameters of the reference layer should be set to 0 so that these parameters will not be updated during the training process.

If you have any questions, please feel free to contact me. My email is wzhshi@hit.edu.cn

If the code is useful to you, please cite our CVPR2019 paper:

@inproceedings{shi2019Scalable,
title={Scalable convolutional neural network for image compressed sensing},
author={Shi, Wuzhen and Jiang, Feng and Liu, Shaohui and Zhao Debin},
booktitle={Computer Vision and Pattern Recognition},
year={2019}
}
