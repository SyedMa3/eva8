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
