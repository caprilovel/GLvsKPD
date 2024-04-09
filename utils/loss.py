import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union

import numpy as np
from einops import rearrange

def group_pattern(n: int, m: int, mat: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
    mat_shape = mat.shape
    assert len(mat_shape) == 2, "The input matrix should be 2D"
    assert mat_shape[0] % n == 0 and mat_shape[1] % m == 0, "The input matrix should be divisible by n and m"
    n1 = mat_shape[0] // n
    m1 = mat_shape[1] // m
    
    mat = rearrange(mat, '(n1 n) (m1 m) -> (n1 m1) (n m)', n=n, m=m, n1=n1, m1=m1)
    return mat
    

def get_group_lasso(model, pattern='dim', *args, **kwargs):
    group_loss = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            if len(param.shape) == 4:
                if pattern == 'dim':
                    param = group_pattern(param.shape[0], param.shape[1], param)
                elif pattern == 'channel':
                    param = group_pattern(param.shape[2], param.shape[3], param)
                else:
                    raise ValueError("Pattern should be either 'dim' or 'channel'")
            if len(param.shape) == 2:
                if pattern == 'dim':
                    param = group_pattern(param.shape[0], param.shape[1], param)
                elif pattern == 'channel':
                    param = group_pattern(param.shape[1], param.shape[0], param)
                    
            group_loss += torch.norm(param, p=2)
            
        
    
    