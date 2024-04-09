import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List
def factorize(n: int, bias=0) -> List[int]:
    # """Return the most average two factorization of n."""
    for i in range(int(np.sqrt(n)) + 1, 1, -1):
        if n % i == 0:
            if bias == 0:
                return [i, n // i]
            else:
                bias -= 1
    return [n, 1]

class KronLinear(nn.Module):
    def __init__(self, in_features, out_features, patchsize=None, shape_bias=0, structured_sparse=False, bias=True, rank_rate=0.1, rank=0) -> None:
        """Kronecker Linear Layer

        Args:
            rank (int): _description_
            a_shape (_type_): _description_
            b_shape (_type_): _description_
            structured_sparse (bool, optional): _description_. Defaults to False.
            bias (bool, optional): _description_. Defaults to True.
        """
        super().__init__()

        if patchsize is not None:
            assert len(patchsize) == 2, "The pathsize should be a tuple of two integers"
            assert in_features % patchsize[0] == 0 and out_features % patchsize[1] == 0, "The input and output features should be divisible by the patchsize"
            a_shape = (in_features // patchsize[0], out_features // patchsize[1])
            b_shape = (patchsize[0], patchsize[1])
        else:
            in_shape = factorize(in_features, shape_bias)
            out_shape = factorize(out_features, shape_bias)
            a_shape = (in_shape[0], out_shape[0])
            b_shape = (in_shape[1], out_shape[1])
        self.rank = rank if rank > 0 else min(a_shape[0], a_shape[1], b_shape[0], b_shape[1]) * rank_rate
        self.rank = int(self.rank) if int(self.rank) > 0 else 1
        
        self.structured_sparse = structured_sparse
        
        if structured_sparse:
            self.s = nn.Parameter(torch.randn( *a_shape), requires_grad=True)
        else:
            self.s = None
        self.a = nn.Parameter(torch.randn(self.rank, *a_shape), requires_grad=True)
        self.b = nn.Parameter(torch.randn(self.rank, *b_shape), requires_grad=True)
        nn.init.xavier_uniform_(self.a)
        nn.init.xavier_uniform_(self.b)
        self.a_shape = self.a.shape
        self.b_shape = self.b.shape
        bias_shape = np.multiply(a_shape, b_shape)
        if bias:
            self.bias = nn.Parameter(torch.randn(*bias_shape[1:]), requires_grad=True)
        else:
            self.bias = None
        
    def forward(self, x):
        # a = self.a
        # if self.structured_sparse:
        #     a = self.s.unsqueeze(0) * self.a
        
        # # a = self.s.unsqueeze(0) * self.a
        # w = kron(a, self.b)
        
        # out = x @ w 
        # if self.bias is not None:
        #     out += self.bias.unsqueeze(0)
        # return out
        # =========================
        a = self.a
        if self.structured_sparse:
            a = self.s.unsqueeze(0) * self.a
        
        # a = self.s.unsqueeze(0) * self.a
        # w = kron(a, self.b)
        x_shape = x.shape 
        b = self.b
        
        x = torch.reshape(x, (-1, x_shape[-1]))
        
        # b = rearrange(b, 'r b1 b2 -> b1 (b2 r)')
        b = b.permute(1, 2, 0).contiguous().view(b.shape[1], -1).contiguous()
        
        # x = rearrange(x, 'n (a1 b1) -> n a1 b1', a1=self.a_shape[1], b1=self.b_shape[1])
        x = x.view(-1, self.a_shape[1], self.b_shape[1]).contiguous()
        
        
        out = x @ b
        
        # out = rearrange(out, 'n a1 (b2 r) -> r (n b2) a1', b2=self.b_shape[2], r=self.rank) 
        out = out.view(-1, self.a_shape[1], self.rank, self.b_shape[2])
        
        # Permute dimensions
        out = out.permute(2, 0, 3, 1)
        out = out.contiguous().view(self.rank, -1, self.a_shape[1]).contiguous()
        
        out = torch.bmm(out, a)
        
        out = torch.sum(out, dim=0).squeeze(0)
        
        
        # out = rearrange(out, '(n b2) a2 -> n (a2 b2)', b2=self.b_shape[2])
        out = out.view(-1, self.b_shape[2], self.a_shape[2]).contiguous()

        # Permute dimensions
        out = out.permute(0, 2, 1)
        # out = out.permute(0, 2, 1).contiguous()

        # Reshape again
        # out = out.view(-1, self.a_shape[2] * self.b_shape[2]).contiguous()
        out = torch.reshape(out, (-1, self.a_shape[2] * self.b_shape[2]))
        
        
        
        out = torch.reshape(out, x_shape[:-1] + (self.a_shape[2] * self.b_shape[2],))
        
        if self.bias is not None:
            out += self.bias.unsqueeze(0)
        return out
    
    
