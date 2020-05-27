# DogBreedClassifier
CNN Project [Udacity Deep Learning Nanodegree]

This is a repository for the Dog Breed Classifier Project in Udacity Deep Learning Nanodegree

It is implemented by using PyTorch library.

-------------------------------------------------------------------------------------------------------------------------------------------


## CNN Model build from scratch
(conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

activation: relu

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

activation: relu

(conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))

activation: relu

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

(pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)

(dropout): Dropout(p=0.3)

(fc1): Linear(in_features=6272, out_features=500, bias=True)

(dropout): Dropout(p=0.3)

(fc2): Linear(in_features=500, out_features=133, bias=True)

Accuracy attained : 16% with 15 epochs

----------------------------------------------------------------------------------------------------------------------------------------

## CNN Model using Tranfer learning
Used **Resnet50** pretrained for transfer learning.

Accuracy attained : 76% with 25 epochs

### Why I Used Resnet50 

I selected Resnet as a pre-trained model because of many of its advantages. One of the biggest advantages of the ResNet is while
increasing network depth, it avoids negative outcomes. So we can increase the depth but we have fast training and higher accuracy.
