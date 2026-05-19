import torch 
import torch.nn as nn
import math 
from dataclasses import dataclass


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
    min_loss : float = 0 # Used only to compare training time

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
        :param k: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=k.device)
        args = k[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding
    
    def forward(self, k : torch.Tensor):
            if k.ndim == 0:
                k = k.unsqueeze(-1)
            t_freq = self.timestep_embedding(k, self.frequency_embedding_size)
            t_emb = self.mlp(t_freq)
            return t_emb


class TimeLinear(nn.Module):
    """MPL"""

    def __init__(self, dim_in: int, dim_out: int, diffusion_steps : int):
        super().__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.diffusion_steps = diffusion_steps

        self.time_embedding = TimeEmbedding(dim_out*2)
        self.fc = nn.Linear(dim_in, dim_out)

    def forward(self, x: torch.Tensor, k: torch.Tensor):
        x = self.fc(x) # <------ Estimate it with close form 
        emb = self.time_embedding(k).view(-1, self.dim_out * 2)
        alpha, beta = emb.chunk(2, dim=-1)
        return (1 + alpha) * x + beta # <--- keep it 


class DenoisingNetTimeGrad(nn.Module):
    """DENOISING NETWORK FOR TIMEGRAD SIMPLE APPLICATION"""

    def __init__(self, in_dim, hidden_size, dim_hids, num_timesteps):
        """
        in_dim : number of time series to noise equals dim_out (since we want to predict the noise on N ts)
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


class TimeGradRNN_LH(nn.Module):
    """ TimeGrad RNN on finite history. h_{t-1} = RNN([x_{t-1}, ..., x_{t-L}])"""
    def __init__(self, input_size, hidden_size, cell = 'LSTM'):
        super().__init__()
        self.in_size = input_size
        self.hidden_size = hidden_size
        if cell == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=1, bias=True, batch_first=True,
                              dropout=0.0, bidirectional=False)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=1, bias=True, batch_first=True,
                               dropout=0.0, bidirectional=False)
        else:
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                              num_layers=1, nonlinearity='tanh', bias=True,
                              batch_first=True, dropout=0.0, bidirectional=False)
        
    def forward(self, x_context : torch.Tensor):
        """Return hidden states history [h_1, ..., h_T] of shape (T, hidden_size)
        x_context : (T,N) tensor
        """
        all_h, h_T = self.rnn(x_context) 
        return all_h[:,-1,:]


class TimeGradRNN_FH(nn.Module):
    def __init__(self, input_size, hidden_size, cell='GRU'):
        super().__init__()
        self.in_size = input_size
        self.hidden_size = hidden_size

        if cell == 'GRU':
            self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size,
                              num_layers=1, bias=True, batch_first=True,
                              dropout=0.0, bidirectional=False)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                               num_layers=1, bias=True, batch_first=True,
                               dropout=0.0, bidirectional=False)
        else:
            self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,
                              num_layers=1, nonlinearity='tanh', bias=True,
                              batch_first=True, dropout=0.0, bidirectional=False)

    def forward(self, x_context):
        all_h, _ = self.rnn(x_context)  
        return all_h





