import torch 
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod 
from EsnUtils.ParametersESN import ParametersESN
from typing import List, Tuple
import math 
from dataclasses import dataclass

    
##############################################################################################################################
# ESN Module
##############################################################################################################################


class ESN(nn.Module):
    """Echo State Network following to Ballarin et Grygoreva (2024)
    s_t = alpha*s_{t-1} + (1-alpha)*sigma(A*s_{t-1} + C*x_t + zeta) $$
    y_{t+1} = a + W^{T}*s_t + e_t 
    """
    def __init__(self, 
                 in_dim :int,
                 reservoir_size : int, 
                 Params : ParametersESN,
                 activ : str = 'tanh',
                 output_layer : str = 'ridge',
                 intercept = False, 
                 alpha_rige = 1.0, 
                 ):
        super().__init__() 
        self.in_dim = in_dim 
        self.reservoir_size = reservoir_size 
        self.leak_rate = Params.leak_rate 
        self.intercept = intercept
        self.alpha_rige = alpha_rige
        self.Params = Params

        self._output_layer(output_layer)
        self._activation_function(activ)
        self._weights()
    
    def _activation_function(self, activ):
        """Define the activation function"""
        if activ == 'sigmoid':
            self.activ = F.sigmoid
        elif activ== 'tanh':
            self.activ = F.tanh
        else:
            raise AttributeError(f"Valid attributes for the activation function: 'tanh', 'sigmoid'. Got {activ}")

    def _output_layer(self,output_layer):

        if output_layer == 'ols':
            self.observation_map = OLS(in_dim=self.reservoir_size,intercept=self.intercept)
        elif output_layer == 'ridge':
            self.observation_map = Ridge(in_dim=self.reservoir_size,intercept=self.intercept, alpha=self.alpha_rige)
        else:
            raise AttributeError(f"Valid attributes for estimation method in the observation map: 'tanh', 'sigmoid'. Got {output_layer}")

    def _weights(self):
        """Sample weights from appropriate laws according to the specification of the ParametersESN class"""
        A, C, zeta = self.Params.sample(self.reservoir_size,self.in_dim)
        self.register_buffer('A', A)
        self.register_buffer('C', C)
        self.register_buffer('zeta', zeta)

    @torch.no_grad()
    def get_states(self, X : torch.Tensor):
        """ 
        Compute reservoir states s_t
        X : (T,N) tensor of inputs
        """
        T = X.shape[0]
        s = torch.zeros(self.reservoir_size, 1) # s_0 (reservoir_state,1)
        states = []
        for t in range(T):
            s = self.leak_rate * s + (1 - self.leak_rate) * self.activ(self.A @ s + self.C @ X[t].reshape(-1,1) + self.zeta)
            states.append(s) # [s_1, ..., s_T]
        S = torch.stack(states)
        self.S = S.squeeze(-1) # (T, reservoir_size)
        return self.S

    @torch.no_grad()
    def fit(self, X : torch.Tensor, y : torch.Tensor):
        """ 
        Fit the observation map
        X : input time series (T,N)
        y : output time series (T,M)
        """
        X = X[:-1] # use X = [x_1, ..., x{T-1}]
        y = y[1:] # use y = [y_2, ..., y_T]
        self.get_states(X=X) # generate the sequence of S
        self.W_out = self.observation_map.fit(X=self.S,y=y) # (reservoir_size,M)
        return self.W_out
    
    def forward(self, X : torch.Tensor):
        S = self.get_states(X=X) # (T,reservoir_size,1)
        y_hat = self.observation_map.predict(S) # should be (T,reservoir_size) * (reservoir_size,M)
        return y_hat, S
    
    @torch.no_grad()
    def update_states(self, X_t: torch.Tensor):
        """
        X_t : (N, 1) tensor
        """
        assert X_t.shape == (self.in_dim, 1), f"Expected shape {(self.in_dim, 1)}, got {X_t.shape}"

        s_prev = self.S[-1].reshape(-1, 1)  # (reservoir_size, 1)

        s_next = (self.leak_rate * s_prev + (1 - self.leak_rate) * self.activ(self.A @ s_prev + self.C @ X_t.reshape(-1, 1) + self.zeta))  # (reservoir_size, 1)

        self.S = torch.cat([self.S, s_next.squeeze(-1).unsqueeze(0)], dim=0)
        return self.S
    
##############################################################################################################################
# Denoising Network
##############################################################################################################################

@dataclass
class TrainSpec:
    """Module to handle training parameters and windows in TimeGrad"""
    t0: int
    Lookback: int
    batch_size: int
    epochs: int
    valid_size : float = 0.2
    patience: int = 30
    lr: float = 1e-3
    print_loss: bool = False

    def __post_init__(self):
        """Method called authom after the init"""
        if self.Lookback > self.t0:
            raise ValueError("Lookback cannot be higher than t0")


class TimeEmbedding(nn.Module):
    """ Time Embedding module from KAIST course in Diffusion Models"""
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(k, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        k: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        dim: the dimension of the output.
        param max_period: controls the minimum frequency of the embeddings.
        return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(-math.log(max_period)* torch.arange(start=0, end=half, dtype=torch.float32)
            / half).to(device=k.device)
        args = k[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, k : torch.Tensor):
            if k.ndim == 0:
                k = k.unsqueeze(-1)
            t_freq = self.timestep_embedding(k, self.frequency_embedding_size)
            t_emb = self.mlp(t_freq)
            return t_emb

class TimeLinear(nn.Module):
    """
    Multi-Layer Perception Network with FiLM 
    """
    def __init__(self, dim_in: int, dim_out: int, diffusion_steps : int):
        super().__init__() 
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.diffusion_steps = diffusion_steps

        self.time_embedding = TimeEmbedding(dim_out*2)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, k: torch.Tensor):
        x = self.fc(x) 
        emb = self.time_embedding(k).view(-1, self.dim_out * 2)
        alpha, beta = emb.chunk(2, dim=-1)
        return (1 + alpha) * x + beta 

class DenoisingNetTimeGrad(nn.Module):
    """DENOISING NETWORK FOR TIMEGRAD SIMPLE APPLICATION"""

    def __init__(self, in_dim : int, hidden_size : int, dim_hids : List, num_timesteps : int):
        """
        in_dim : number of time series to noise equals dim_out (since we want to predict the noise on N ts)
        dim_hids : list of size of th
        """
        super().__init__()
        
        # After cat([x, h], dim=-1): input dim = 1 + hidden_size
        self.dim_in = in_dim + hidden_size
        dim_out = in_dim
        
        dims = [self.dim_in] + dim_hids + [dim_out]  
        self.tlins = nn.ModuleList([
            TimeLinear(dims[i], dims[i+1], num_timesteps)
            for i in range(len(dims) - 1)
        ]) # <------ separa the last linear here 
        self.act = nn.SiLU()

    def forward(self, x, k, h):
        """ Predict the Noise using the noisy input, the diffusion step and the hidden state"""
        x = torch.cat([x, h], dim=-1)   
        for i, layer in enumerate(self.tlins):
            x = layer(x, k)
            if i < len(self.tlins) - 1:
                x = self.act(x)
        return x



#########################################################################################################################
# Not needed for the Diffusion-ESN
#########################################################################################################################

class ObservationMap(ABC):

    def __init__(self, in_dim : int, intercept: bool = False):
        self.intercept = intercept
        self.in_dim = in_dim    

    def _intercept(self, X: torch.Tensor) -> torch.Tensor:
        """Prepend a column of ones to X (same device & dtype as X)."""
        bias = torch.ones(X.shape[0], 1, dtype=X.dtype, device=X.device)
        return torch.cat([bias, X], dim=1)
    
    @abstractmethod
    def fit(self, X : torch.Tensor, y : torch.Tensor):
        pass 

    @abstractmethod
    def predict(self, X : torch.Tensor):
        pass

class OLS(ObservationMap):
    """ 
    OLS estimation of the output layer: 
        W = (X'X)^-1 X'y
    """
    def __init__(self, in_dim : int, intercept: bool = False):
        super().__init__(in_dim=in_dim, intercept=intercept)
        self.W = None

    @torch.no_grad()
    def fit(self, X : torch.Tensor, y : torch.Tensor):
        """
        X : (T,N)
        y : (T, M)
        """
        assert X.shape[0] == y.shape[0], "The number of time period must be equal for X and y"

        if self.intercept:
            X = self._intercept(X=X)

        inv = torch.linalg.inv(X.T @ X)
        self.W = inv @ X.T @ y # (N,M)
        return self.W 

    @torch.no_grad()
    def predict(self, X : torch.Tensor):
        if self.W is None:
            raise RuntimeError("Call fit() before forward()")
        if self.intercept:
            X = self._intercept(X=X)
        return torch.matmul(X,self.W)


class Ridge(ObservationMap):
    """
    Ridge regression: y = X @ W + e: min_W  ||y - X @ W||^2_2 + alpha * ||W||^2_2
    Implementation according to Ballarin et al. (2025)
    Idea for the code from: https://gist.github.com/myazdani/3d8a00cf7c9793e9fead1c89c1398f12
    """
    def __init__(self, in_dim : int, alpha: float = 1.0, intercept: bool = False):
        super().__init__(in_dim=in_dim, intercept=intercept)
        assert alpha >= 0, "alpha must be non-negative"
        self.alpha = alpha
        self.W_ridge = None

    @torch.no_grad()
    def fit(self, X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Fit the ridge estimator.
        X : (T, N)
        y : (T, M)
        """
        assert X.shape[-1] == self.in_dim, f"Mispecified number of X features: in_dim = {self.in_dim}, got {X.shape[-1]}"
        assert X.shape[0] == y.shape[0], "X and y must have the same number of rows"

        if self.intercept:
            X = self._intercept(X)      

        N = X.shape[1]
        T = X.shape[0]

        I = torch.eye(N, dtype=X.dtype, device=X.device)
        if self.intercept:
            I[0, 0] = 0.0  # don't regularize intercept
           

        # W = (X'X + alpha*(T-1)*I)^{-1} X'y
        S = torch.linalg.inv((X.T @ X) + self.alpha*(T-1)*I)
        self.W_ridge = S @ X.T @ y # Should be (N,M)

        return self.W_ridge 

    @torch.no_grad()
    def predict(self, X: torch.Tensor):
        """
        Predict:  ŷ = X @ W_ridge
        X : (T,N)
        """
        if self.W_ridge is None:
            raise RuntimeError("Call fit() before forward()")

        if self.intercept:
            X = self._intercept(X)

        return torch.matmul(X,self.W_ridge)