# Documentation

In this notebook, our objective is to create a neural network which can learn from the MNIST Dataset and predict the labels with 99.4% accuracy in less than 20k parameters and in 20 epochs.

## Network

First, we define out neural netwrok `Net`.

The layers for the network are:

```python
Conv2d(1, 8, 3, padding=1)
Conv2d(8, 8, 3, padding=1)
MaxPool2d(2, 2)
Conv2d(8, 16, 3)
Conv2d(16, 16, 3)
MaxPool2d(2, 2)
Conv2d(16, 32, 3)
Conv2d(32, 32, 3)
AvgPool2d((1,1))
Linear(32, 10)
```

> We use Dropout, ReLU, Batch Normalisation after every convolution layer(except in the last we use only ReLU)

### Summary of our network

Using `torchsummary` we can see the summary of our network

```python
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 8, 28, 28]              80
           Dropout-2            [-1, 8, 28, 28]               0
       BatchNorm2d-3            [-1, 8, 28, 28]              16
            Conv2d-4            [-1, 8, 28, 28]             584
           Dropout-5            [-1, 8, 28, 28]               0
       BatchNorm2d-6            [-1, 8, 28, 28]              16
         MaxPool2d-7            [-1, 8, 14, 14]               0
            Conv2d-8           [-1, 16, 12, 12]           1,168
           Dropout-9           [-1, 16, 12, 12]               0
      BatchNorm2d-10           [-1, 16, 12, 12]              32
           Conv2d-11           [-1, 16, 10, 10]           2,320
          Dropout-12           [-1, 16, 10, 10]               0
      BatchNorm2d-13           [-1, 16, 10, 10]              32
        MaxPool2d-14             [-1, 16, 5, 5]               0
           Conv2d-15             [-1, 32, 3, 3]           4,640
          Dropout-16             [-1, 32, 3, 3]               0
      BatchNorm2d-17             [-1, 32, 3, 3]              64
           Conv2d-18             [-1, 32, 1, 1]           9,248
        AvgPool2d-19             [-1, 32, 1, 1]               0
           Linear-20                   [-1, 10]             330
================================================================
Total params: 18,530
Trainable params: 18,530
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.40
Params size (MB): 0.07
Estimated Total Size (MB): 0.47
----------------------------------------------------------------
```

As we can see, our parameters are < 20k.

One should also note that the estimated total size of our network is only 0.47 MB. And we will see that this network predict with very high accuracy. The power of deep learning.

## Training

