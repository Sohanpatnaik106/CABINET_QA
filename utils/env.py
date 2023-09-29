# Import the required libraries
import torch
import numpy as np
import random as rn

# Set global seed
def set_seed(seed):
    rn.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True