PANDAS_RANDOM_STATE = 10
PYTORCH_SEED = 12
RANDOM_SEED = 22
NUMPY_SEED = 42

import torch
import random
import numpy as np

torch.manual_seed(PYTORCH_SEED)
random.seed(RANDOM_SEED)
np.random.seed(NUMPY_SEED)