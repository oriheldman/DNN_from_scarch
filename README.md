# DNN_from_scarch
A neural network “from scratch” including: Forward/Backword propagation, batch normalization and dropout.
Evaluation prediction task: classify the MNIST dataset (grayscale digit images) 

## NN Architecture
The NN consists of 4 layers, aside from the input layer. The layers are the following size: 20, 7, 5 ,10 (final layer is the output layer). Each layer is a fully connected layer (FC) that uses the Relu activation function in the hidden layers and the softmax function in the output layer. The NN supports batch normalization and dropout.
As part of its implementation, the NN holds a “Train_mode” boolean parameter which note the state of the NN (iterating over data with updating parameters or provide predictions for new samples). The “Train_mode” parameter is being used in the batch normalization and the dropout applications because there are different behaviors during the training and the predicting phases.

## Cost function
The cost function of the NN is the categorical cross-entropy loss, cost=-1/m*∑_1^m▒∑_1^C▒〖y_i  log⁡(y ̂ ) 〗, where y_i is the ground truth (onehot vector for the correct class) and y ̂ is the softmax-adjusted prediction.

## NN Parameters Initialization 
The weight matrices are randomly initialized with values between 0 and 1. In addition, these random values are multiplied by a small scalar of 0.01 to make the activation units active and be on the regions where activation functions’ derivatives are not close to zero.
The bias vectors are initialized to zero vectors.

## Batch normalization support
The NN supports batch normalization (BN). If batch normalization is enabled, BN will be applied after each hidden layer. The following describes the different behaviors according to the NN mode (train or test):
•	Train_mode - Normalize the output of each layer (except last) after the activation function. Furthermore, We save the batch_mean and batch_std of each neuron in the layer.
•	Test_mode - Since we are not allowed to normalize with other test samples, we normalize the neurons output of each layer with the corresponding mean and std saved during the training stage.

## Dropout
When Dropout is enabled, it will be applied to the output of each neuron, in addition to the input layer. The following describes the different behaviors according to the NN mode (train or test):
•	Train_mode -  given a dropout_ratio (hardcoded probability from 0 to 1) – mask each neuron output in the input/layers with 0 in probability of (1 -  dropout_ratio). Then, Multiply other neurons (the activated ones) with (1 / (1 -dropout_ratio) ).
•	Test_mode – The Dropout is off during test mode.

## Early stopping criterion:
We Trained the NN with maximum of 50k iterations. During the train we compute the validation loss, and save the min validation loss achieved. Furthermore, we count the number of iterations in a row without improvement on the validation loss. The calculation occur every 5 iterations for speed optimization reasons. If we reach the limit of max iterations (was set to 500 for convergence reasons) without improvement we stop the training.

# Prediction Tasks
The prediction task is a multi-class classification task on the mnist dataset. The NN was trained in the following configurations:
1.	Without using batch normalization and no dropout - baseline
2.	With batch normalization and no dropout – baseline + BN
3.	Without using batch normalization and dropout of 10% - baseline + Dropout
4.	With batch normalization and dropout of 10% - baseline + BN + Dropout
All configurations used a learning rate of 0.009

## The Mnist Dataset
The mnist is a dataset of 60,000 square 28×28 pixel grayscale images of handwritten single digits between 0 and 9. The dataset is split into train-test sets, where 80% for the train set and 20% for test set.  From the train set we randomly take 20%  samples for creating validation set.
Data Preparation and Preprocessing
The images are flattened to a 1d vector of length h*w, hence each instance of the mnist dataset (grayscale 28X28) image is a 784 vector.
minmax normalization used to the input data of the network. As a result, the input layer receives values between 0-1.
