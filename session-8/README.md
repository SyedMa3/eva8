# Assignment 8

## Objective

- Obtain ~90% accuracy on CIFAR10
- Using a custom ResNet-like architecture
- Use OneCyclePolicy
- Use albumentations library
- Code should modular

## Solution

### Network Architecture

```
1. PrepLayer - (Conv 3x3 s1, p1) >> BN >> RELU [64k]
2. Layer1 -
    a. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
    b. R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k] 
    c. Add(X, R1)
3. Layer 2 -
    a. Conv 3x3 [256k]
    b. MaxPooling2D
    c. BN
    d. ReLU
4. Layer 3 -
    a. X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
    b. R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
    c. Add(X, R2)
5. MaxPooling with Kernel Size 4
6. FC Layer 
7. SoftMax
```

No. of Parameters - `6,573,130`

### Image Augmentations used

- `RandomCrop(32,32) after padding of 4`
- `horizontal flip`
- `coarseDropout (max_holes = 1, max_height=8px, max_width=8px, min_holes = 1, min_height=8px, min_width=8px, fill_value=(mean of dataset), mask_fill_value = None)`

### OneCyclePolicy Stats

- MaxLR = 0.01
- Epochs = 24
- pct_start = 0.15
- final_div_factor=10

### Accuracy

- Training Accuracy - 93.93%
- Testing Accuracy - 90.02%
