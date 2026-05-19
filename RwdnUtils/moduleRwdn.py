import torch 
import torch.nn as nn 
from RwdnUtils.parametersRwdn import ParametersRWDN



class RWDenoisingNetwork(nn.Module): 

    methods = ['OLS', 'Inverse', 'Ridge']
    activations = ['SiLU', 'ReLU', 'tanh', 'sigmoid']
    nn_activ = {'SiLU' : nn.SiLU, 'ReLU': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid' : nn.Sigmoid}


    def __init__(self, N : int, H : int, K : int, S: int, params : ParametersRWDN, activ : str = 'SiLU', lam = 1e-3):
        """ 
        N       : (int) number of features in x_k 
        H       : (int) number of neurons in the Hidden State of the Denoising Network 
        K       : (int) number of diffusion steps 
        S       : (int) size of hidden state of the RNN or the ESN
        params  : ParametersRWDN instance to sample the Random Weights 
        activ   : (str) activation function 
        """
        super().__init__()
        self.A, self.b = params.sample(K=K,hidden_size=H,in_size=N+S)
        self.N = N 
        self.H = H 
        self.K = K 
        self.S = S
        # Define W as I in the constructor such that the forward gives the randomProjection
        self.W = torch.eye(n=H,m=N).unsqueeze(0).expand(K, -1, -1).clone()
        self.lam = lam
        self.activation_fucntion(activ)

    def activation_fucntion(self, activ): 
        assert activ in self.activations, f"Available activations: {self.activations}, got {activ}"
        self.sigma = self.nn_activ[activ]()

    @torch.no_grad()
    def fit(self, X_k, H_t, E_k, k, method = "Ridge"):
        """ 
        X_k : matrix of the noised process (T,N)
        H_t : matrix of hidden state from the RNN or ESN (T, S)
        E_k : matrix of noise (T,N)
        k   : diffusion step int
        """
        assert method in self.methods, f"Method not available. Possible {self.methods} got {method}"

        U = torch.cat([X_k, H_t], dim = 1)

        Z = self.sigma(U @ self.A[k].T + self.b[k].squeeze(1)) # (T,H)

        if method == 'OLS':
            self.W[k] = torch.linalg.inv(Z.T @ Z) @ Z.T @ E_k #(H,N)
        elif method == 'Inverse':
            self.W[k] = Z.T @ torch.linalg.inv(Z @ Z.T) @ E_k #(H,N)
            print(self.W[k].shape)
        elif method == 'Ridge':
            lam = 1e-3  # ridge regularization
            self.W[k] = torch.linalg.solve(Z.T @ Z + self.lam * torch.eye(self.H), Z.T @ E_k)

    @torch.no_grad()
    def forward(self, x_k, h_t, k):
        """
        x_k : (T, N) or (1, N)
        h_t : (T, S) or (1, S)  — broadcast if needed
        """
        # Broadcast h_t if it's a single state but x_k has T samples
        if h_t.shape[0] == 1 and x_k.shape[0] > 1:
            h_t = h_t.expand(x_k.shape[0], -1)   # (T, S)

        u           = torch.cat([x_k, h_t], dim=1)                    # (T, N+S)
        z           = self.sigma(u @ self.A[k].T + self.b[k].squeeze(1))  # (T, H)
        epsilon_hat = z @ self.W[k]                                    # (T, N)
        return epsilon_hat