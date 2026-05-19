import torch
import torch.nn as nn


def draw(shape: tuple[int, int], dist: str, sparsity: float = 0.0, norm_type: str = 'spectral_radius') -> torch.Tensor:
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

class RandomWeight:
    """
    Class for defining the distribution of the parameters, sparsity and normalization
    """
    VALID_DISTS = ('normal', 'uniform', 'zeros')
    VALID_NORMS = (None, 'spectral_radius', 'l1', 'l2', 'linf', 'euclidean')

    def __init__(self, dist: str, sparsity: float, norm_type: str):
        assert dist in self.VALID_DISTS, f"Invalid dist '{dist}'. Choose from {self.VALID_DISTS}"
        assert norm_type in self.VALID_NORMS, f"Invalid norm '{norm_type}'. Choose from {self.VALID_NORMS}"
        assert 0.0 <= sparsity < 1.0, f"sparsity must be in [0, 1), got {sparsity}"
        self.dist = dist
        self.sparsity = sparsity
        self.norm_type = norm_type

    def __repr__(self):
        return f"RandomWeight(dist='{self.dist}', sparsity={self.sparsity}, norm_type='{self.norm_type}')"



class ParametersESN(nn.Module):
    """
    Parameters of ESN:
        leak_rate   : leak rate of the ESN
        rho         : spectral radious of the reservois matrix
        gamma       : input shifting
        omega       : bias shifting
    """
    def __init__(self,
                 A_RW    : RandomWeight,
                 C_RW    : RandomWeight,
                 zeta_RW : RandomWeight,
                 leak_rate : float,
                 rho       : float,
                 gamma     : float,
                 omega     : float, 
                 seed      : int = 12345):
        super().__init__()
        self.A_RW     = A_RW
        self.C_RW     = C_RW
        self.zeta_RW  = zeta_RW
        self.leak_rate = leak_rate
        self.rho      = rho
        self.gamma    = gamma
        self.omega    = omega
        self.seed     = seed
        self.check_params()

    def check_params(self):
        assert 0.0 < self.rho < 1.0,    "Spectral radius rho must be in (0, 1)"
        assert self.omega >= 0.0,        "Shift scaling omega must be >= 0"
        assert self.gamma >= 0.0,        "Input scaling gamma must be >= 0"
        assert 0.0 <= self.leak_rate <= 1.0, "leak_rate must be in [0, 1]"

    def __repr__(self):
        return (f"ParametersESN(rho={self.rho}, gamma={self.gamma}, "
                f"omega={self.omega}, leak_rate={self.leak_rate})")
    
    def sample(self, reservoir_size, in_size):
        """Sample bar{A},bar{C},bar{zeta} and return A,C,zeta from appropriate laws"""
        torch.manual_seed(self.seed)
        A_bar = draw(shape=(reservoir_size,reservoir_size), dist = self.A_RW.dist, sparsity = self.A_RW.sparsity, norm_type = self.A_RW.norm_type)
        C_bar = draw(shape=(reservoir_size,in_size), dist = self.C_RW.dist, sparsity = self.C_RW.sparsity, norm_type = self.C_RW.norm_type)
        zeta_bar = draw(shape=(reservoir_size,1), dist = self.zeta_RW.dist, sparsity = self.zeta_RW.sparsity, norm_type = self.zeta_RW.norm_type)

        A = self.rho*A_bar
        C = self.gamma*C_bar 
        zeta = self.omega*zeta_bar 

        return A,C,zeta