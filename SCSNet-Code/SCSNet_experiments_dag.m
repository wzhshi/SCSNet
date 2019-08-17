%% Experiment with the cnn_mnist_fc_bnorm
clc
close all

[net_bn, info_bn] = SCSNet_dag(...
  'expDir', './results');
