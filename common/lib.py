import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import torch
from torch.utils.data import TensorDataset  # group arr of input and arr of labels to a tuple ([inputs], [labels])
from torch.utils.data import Dataset, DataLoader, ConcatDataset # split the data into batches of a predefined size while training. It also provides other utilities like shuffling and random sampling of the data.
import os
from sklearn.model_selection import KFold
from scipy.stats import wilcoxon
from torch.autograd import Function
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
import scipy
import pandas as pd
import random
import math
import sys
import torch.nn.functional as F  # Parameterless functions, like (some) activation functions
from torch import optim  # For optimizers like SGD, Adam, etc.
from tqdm import tqdm  # For nice progress bar!
from numba import jit
from functools import partial
from dataclasses import dataclass
from collections import OrderedDict
import time
from sklearn.metrics import confusion_matrix, classification_report
np.set_printoptions(threshold=sys.maxsize)
torch.set_printoptions(threshold=sys.maxsize)

data_dir = "./ucr"
result10_dir = "./Results10"
result_dir = "./Results"
plot_dir = "./Plots"