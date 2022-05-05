DTW Neural Networks for Time Series Classification
===
Based on [Elastic Softmax Regression](https://github.com/bjjain/jESMR) with Deep Neural Network integration capapbilities.

## Getting Started
### Requirements:
```
numpy
seaborn
pytorch
scikit-learn
tqdm
numba==0.53
```
This code mainly depends on [PyTorch](https://pytorch.org/) and [Numba](http://numba.pydata.org/).
### Training
There are 2 main training files:  and ```train10.py```.
+ ```train.py```: Contain all trainable models ( Basic ESMR, Mock ESMR, Base FCN, Base ResNet1D, and their variation with ESMR integration)
+ ```train10.py```: Used to compare 10 runs between 2 different models
### Plotting
In addition to 2 plotting related files:
+ ```plot_data.py```: Used to plot all training and testing Time Series datasets
+ ```visualized.py```: Used to inspect the trained weights of ESMR (WarpReg) layer

## Example Usage
```python train.pu```
