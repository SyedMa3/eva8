# Session 2.5

This file contains the documentation for session 2.5 assignment.

## Data Representation

Since this problem contains two types of input which we have to combine we define a custom Dataset using `torch.utils.data.Dataset`

```python
class AddDataset(Dataset):
  """
    The dataset will return a tuple ((image, label), random number)
  """
  def __init__(self, mnist_set):
    self.data = mnist_set

  def __getitem__(self, index):
    r = self.data[index]
    image, label = r
    n = np.random.randint(10) #function used to create a random integer in [0,9]

    return (image, label) , n

  def __len__(self):
    return len(self.data)
```

Our custom dataset(`AddDataset`), returns a tuple `((image, label), random number)`

### Data generation

For creating a random number in [0,9] we are using the method `numpy.random.randint`.

## Network

We divided our network into two "blocks". The first one deals with the MNIST image and predicts the label. The second one takes the prediction from before and also the random number generated and tries to solve the addition problem.

## Evaluation

We used CrossEntropy Loss since this is a case of multi-class classification.

After training, we got satisfactory results. And we can see from training logs, the accuracy for image is more and it learns quickly for MNIST compared to a simple sum function.

> For evaluation, we could have used MNIST test dataset, but since the objective of this assignment was to learn how to define a custom dataset and neural network, we just used the train dataset for measuring accuracy, which is in general a bad way for evaluation.

## Training logs

```python
epoch 0
image loss: 0.06495117396116257 add loss: 0.6108326315879822 image_accuracy: 0.9348833333333333 add_accuracy: 0.6464833333333333
epoch 1
image loss: 0.029963163658976555 add loss: 0.19929839670658112 image_accuracy: 0.9765833333333334 add_accuracy: 0.9352333333333334
epoch 2
image loss: 0.17872925102710724 add loss: 0.5085540413856506 image_accuracy: 0.9787333333333333 add_accuracy: 0.9696666666666667
epoch 3
image loss: 0.07893143594264984 add loss: 0.2554011344909668 image_accuracy: 0.9832166666666666 add_accuracy: 0.9791333333333333
epoch 4
image loss: 0.04043226316571236 add loss: 0.11252318322658539 image_accuracy: 0.9825833333333334 add_accuracy: 0.9801
epoch 5
image loss: 0.08602919429540634 add loss: 0.37573131918907166 image_accuracy: 0.9833833333333334 add_accuracy: 0.9761666666666666
epoch 6
image loss: 0.09629198908805847 add loss: 0.47857797145843506 image_accuracy: 0.9848166666666667 add_accuracy: 0.9817333333333333
epoch 7
image loss: 0.09620197117328644 add loss: 0.24458816647529602 image_accuracy: 0.9839833333333333 add_accuracy: 0.9813
epoch 8
image loss: 0.06737439334392548 add loss: 0.36570143699645996 image_accuracy: 0.9845333333333334 add_accuracy: 0.9807666666666667
epoch 9
image loss: 0.07602549344301224 add loss: 0.20091837644577026 image_accuracy: 0.9854833333333334 add_accuracy: 0.9819166666666667
epoch 10
image loss: 0.04324892908334732 add loss: 0.12059669196605682 image_accuracy: 0.9870666666666666 add_accuracy: 0.9855
epoch 11
image loss: 0.030623624101281166 add loss: 0.22408565878868103 image_accuracy: 0.9883666666666666 add_accuracy: 0.9851666666666666
epoch 12
image loss: 0.016037501394748688 add loss: 0.07988319545984268 image_accuracy: 0.98675 add_accuracy: 0.9812
epoch 13
image loss: 0.019993547350168228 add loss: 0.01991249807178974 image_accuracy: 0.9875 add_accuracy: 0.9863833333333333
epoch 14
image loss: 0.09010564535856247 add loss: 0.2202335149049759 image_accuracy: 0.9865833333333334 add_accuracy: 0.9841833333333333
```
