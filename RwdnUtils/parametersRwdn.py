import torch 
import torch.nn as nn
from typing import Tuple
from EsnUtils.ParametersESN import RandomWeight

def draw(shape: Tuple[int, int, int], dist: str, sparsity: float = 0.0, norm_type: str = 'spectral_radius') -> torch.Tensor:
    """
    Draw a random tensor form choosen law, with selected degree of sparsity and normalization
    """

    if dist == 'normal':
        M = torch.randn(size=shape)
    elif dist == 'uniform':
        M = torch.empty(size=shape).uniform_(-1.0, 1.0)
    elif dist == 'zeros':
        M = torch.zeros(size=shape)
    else:
        raise ValueError(f"Invalid dist '{dist}'. Choose from ('normal', 'uniform', 'zeros')")

    # Sparsity
    if sparsity > 0:
        mask = torch.bernoulli(torch.full_like(M, 1.0 - sparsity))
        M = M * mask

    # Normalization
    if norm_type is None:
        return M
    elif norm_type == 'spectral_radius':
        rho = torch.linalg.eigvals(M).abs().max()
        if rho == 0.0:
            raise RuntimeError("Spectral radius is 0 after sparsification. Reduce sparsity or change seed.")
        return M / rho
    elif norm_type in ('l1', 'l2', 'linf'):
        ord_ = {'l1': 1, 'l2': 2, 'linf': float('inf')}[norm_type]
        col_norms = torch.linalg.norm(M, ord=ord_, dim=0, keepdim=True).clamp(min=1e-8)
        return M / col_norms
    elif norm_type == 'euclidean':
        return M / torch.linalg.norm(M).clamp(min=1e-8)
    else:
        raise NotImplementedError("Normalization is not implemented")


class ParametersRWDN(nn.Module):
    """
    Parameters of RWDN:
        gamma       : input shifting
        omega       : bias shifting
    """
    def __init__(self,
                 A_k    : RandomWeight,
                 b_k    : RandomWeight,
                 gamma     : float,
                 omega     : float, 
                 seed      : int = 12345):
        super().__init__()
        self.A_k     = A_k
        self.b_k     = b_k
        self.gamma    = gamma
        self.omega    = omega
        self.seed     = seed
    
    def sample(self, K, hidden_size, in_size) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample bar{A},bar{C},bar{zeta} and return A,C,zeta from appropriate laws"""
        torch.manual_seed(self.seed)
        A = draw(shape=(K, hidden_size,in_size), dist = self.A_k.dist, sparsity = self.A_k.sparsity, norm_type = self.A_k.norm_type)
        b = draw(shape=(K, hidden_size,1), dist = self.b_k.dist, sparsity = self.b_k.sparsity, norm_type = self.b_k.norm_type)

        A = self.gamma*A 
        b = self.omega*b 

        return A,b