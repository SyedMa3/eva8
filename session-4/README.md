# Session 4 Assignment

Goal - Achieve 99.4 accuracy on MNIST dataset, under 10k parameters and 15 epochs.

We have divided our attempt into three iterations of code.

## Summary

### Iteration 1

[Notebook](./session_4_1.ipynb)

### Target

- Have a basic working code
- Have a skeleton code to build upon
- Make it as light as possible without adding any normalisation, regularisation, etc.

### Results

- Parameters: 9.6k
- Best Train accuracy: 99.15
- Best Test accuracy: 98.85

### Analysis

- Good model
- No overfitting
- Can be better if trained more

---

### Iteration 2

[Notebook](./session_4_2.ipynb)

### Target

- Add normalisation, regularisation, GAP.
- Increase the capacity after adding GAP, but not too much.
- Make it as light as possible without adding any normalisation, regularisation, etc.

### Results

- Parameters: 7.1k
- Best Train accuracy: 98.78
- Best Test accuracy: 99.30

### Analysis

- Normalisatoin, Dropout, GAP working.
- Highly potent model even with less parameters.
- Currently underfitting. Can be improved by training more.

---

### Iteration 3

[Notebook](./session_4_3.ipynb)

Combined Code 8,9 from the lecture

### Target
- Add image augmentation.
- Add LR scheduler.
- 
### Results

- Parameters: 9.9k
- Best Train accuracy: 98.91
- Best Test accuracy: 99.48

### Analysis

- Model under-fitting, hence augmentation works
- Achieved high accuracy faster, but stabilised around 99.4
- Can achieve more accuracy if trained more.