from TimeGradUtils.diffModules import BaseScheduler
from RwdnUtils.moduleRwdn import RWDenoisingNetwork
from EsnUtils.ModuleESN import ESN
from tqdm import tqdm
import torch
import torch.nn as nn

class DDPM_RWDN(nn.Module):
    def __init__(self, scheduler : BaseScheduler):
        super().__init__()
        self.scheduler = scheduler

    @torch.no_grad()
    def q_sample(self, x_0 : torch.Tensor, k: torch.Tensor, noise = None):
        r"""
        Sample x_t ~ q(x_t | x_0) using the reparameterization trick.
        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1 - ᾱ_t) * ε
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_bar_k = self.scheduler.alpha_bar[k].view(-1, 1)

        x_t = torch.sqrt(alpha_bar_k) * x_0 + torch.sqrt(1 - alpha_bar_k) * noise

        return x_t, noise

    @torch.no_grad()
    def p_sample(self,model: RWDenoisingNetwork, x_k: torch.Tensor, h_t: torch.Tensor, k: int, variance = 'beta'):
        """
        Sample from the reverse process p(x_{k-1} | x_k) ~ N(mu, sigma)
        x_k : (n_samples, N)
        h_t : (n_samples, S)  — ESN hidden state at this step
        """
        eps = model(x_k=x_k, h_t=h_t, k=k)        # (n_samples, N)

        alpha_k          = self.scheduler.alpha[k]
        alpha_bar_k      = self.scheduler.alpha_bar[k]
        beta_k           = self.scheduler.betas[k]

        mu = (1 / torch.sqrt(alpha_k)) * (x_k - (beta_k / torch.sqrt(1 - alpha_bar_k)) * eps)

        if k > 0:
            if variance == 'beta':
                sigma_k          = torch.sqrt(beta_k)
            elif variance == 'beta_tilde':
                alpha_bar_k_prev = self.scheduler.alpha_bar[k-1]
                beta_tilde_k     = (1-alpha_bar_k_prev) / (1-alpha_bar_k) * beta_k
                sigma_k          = torch.sqrt(beta_tilde_k)
            else: 
                alpha_bar_k_prev = self.scheduler.alpha_bar[k-1]
                beta_tilde_k     = (1-alpha_bar_k_prev) / (1-alpha_bar_k) * beta_k
                sigma_k          = torch.sqrt(beta_tilde_k)
            x_prev           = mu + sigma_k * torch.randn_like(x_k)
        else:
            x_prev = mu

        return x_prev


    @torch.no_grad()
    def reverse_sampling(self,
            x             : torch.Tensor,# (T, N) clean conditioning sequence
            model         : RWDenoisingNetwork,
            esn           : ESN,
            n_samples     : int,
            steps_to_plot : list = [], 
            variance : str = 'beta'):

        # Run ESN on clean data, take last hidden state as context
        s   = esn.get_states(X=x)               # (T, S)
        s_t = s[-1].unsqueeze(0)               # (1, S) since broadcast in forward, to fix

        x_rev = torch.randn((n_samples, model.N))  # x_K ~ N(0, I)
        x_selected_steps = []

        for k in reversed(range(model.K)):
            x_rev = self.p_sample(model, x_k=x_rev, h_t=s_t, k=k, variance=variance)

            if k in steps_to_plot:
                x_selected_steps.append(x_rev.clone())

        return x_rev, x_selected_steps
    
    @torch.no_grad()
    def sample_trajectory(self, DeNetmodel: RWDenoisingNetwork, esn: ESN, x: torch.Tensor,
                        n_steps: int = 500, n_samples: int = 1):

        contexts = x.clone().unsqueeze(0).expand(n_samples, -1, -1).clone()  # (n_samples, T, N)
        generated = []

        for _ in tqdm(range(n_steps), desc="Generation", leave=False):

            x_t_list = []
            for i in range(n_samples):
                x_t_i, _ = self.reverse_sampling(x=contexts[i],
                                            esn=esn, model=DeNetmodel,
                                            n_samples=1)            # (1, N)
                x_t_list.append(x_t_i)

            x_t = torch.cat(x_t_list, dim=0)                       # (n_samples, N)
            generated.append(x_t)

            # Each sample slides its own context
            x_t_ctx = x_t.unsqueeze(2)                             # (n_samples, N, 1) ... still to fix...
            contexts = torch.cat([contexts[:, 1:, :], x_t.unsqueeze(1)], dim=1)  # (n_samples, T, N)

        return torch.stack(generated, dim=1)  # (n_samples, n_steps, N)