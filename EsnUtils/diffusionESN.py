import torch
import torch.nn as nn
import numpy as np 
from typing import Tuple, List
from tqdm import tqdm 
from tqdm import tqdm 
from EsnUtils.ModuleESN import ESN

class BaseScheduler(nn.Module):
    """
    Variance scheduler of DDPM.
    """
    def __init__(
        self,
        num_train_timesteps: int,
        beta_1: float = 1e-4,
        beta_K : float = 0.1, # Rasul et al. from 1*10^-4 to 0.1 (different from DDPM)
        s : float = 0.008,
        mode: str = "linear",
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.timesteps = torch.from_numpy(
            np.arange(0, self.num_train_timesteps)[::-1].copy().astype(np.int64)
        )

        if mode == "linear":
            betas = torch.linspace(beta_1, beta_K, steps=num_train_timesteps)
        elif mode == "quad":
            betas = (torch.linspace(beta_1**0.5, beta_K**0.5, num_train_timesteps) ** 2)
        elif mode == "cosine":
            """Cosine schedule from Imporved DDPM"""
            k = torch.arange(0, num_train_timesteps + 1).float()
            f_k = torch.cos((k / num_train_timesteps + s) / (1 + s) * torch.pi / 2) ** 2
            alpha_bar = f_k / f_k[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = torch.clip(betas, 0.0001, 0.999)

        else:
            raise NotImplementedError(f"{mode} is not implemented.")

        alphas = torch.ones(betas.shape) - betas
        alphas_cumprod = torch.cumprod(alphas, dim = 0)

        # use register_buffer for cuda 
        self.register_buffer("betas", betas)
        self.register_buffer("alpha", alphas)
        self.register_buffer("alpha_bar", alphas_cumprod)



class DDPM:
    """DDPM main class"""
    def __init__(self, scheduler : BaseScheduler):
        self.scheduler = scheduler

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
    def p_sample(self, model, x_t: torch.Tensor, k : int, y : torch.Tensor):
        r"""
        Sample from the reverse  p(x_{k-1} | x_t) ~ N(mu,sigma)
        """
        t_tensor = torch.tensor([k], device=x_t.device).expand(x_t.shape[0])
        eps = model(x_t, t_tensor, y=y)
        
        alpha_k     = self.scheduler.alpha[k]
        alpha_bar_k = self.scheduler.alpha_bar[k]
        beta_k      = self.scheduler.betas[k]
        alpha_bar_k_prev = self.scheduler.alpha_bar[k-1] # \bar{\alpha_{k-1}} 

        mu = (1 / torch.sqrt(alpha_k)) * (x_t - (beta_k / torch.sqrt(1 - alpha_bar_k)) * eps)

        if k > 0:
            z = torch.randn_like(x_t)
            sigma_k = torch.sqrt(beta_k)
            # or use
            beta_tilde_k = (1-alpha_bar_k_prev)/(1-alpha_bar_k)*beta_k
            sigma_k = torch.sqrt(beta_tilde_k)
            x_prev = mu + sigma_k * z
        else:
            x_prev = mu 

        return x_prev

    @torch.no_grad()
    def reverse_sampling_fl(self, model, n_samples: int, y: float, steps_to_plot : list = []):
        """
        Full reverse process according to DDPM conditioning on a scalar y
        """
        x = torch.randn((n_samples, 1))
        y_tensor = torch.full((n_samples, 1), y)
        x_selected_steps = []
        
        for k in reversed(range(self.scheduler.num_train_timesteps)):
            x = self.p_sample(model, x_t=x, k=k, y=y_tensor)
            if k in steps_to_plot:
                x_selected_steps.append(x)
        
        return x, x_selected_steps


class ESN_DDPM(DDPM):
    """DDPM with RNN conditioning for TimeGrad"""
    @torch.no_grad()
    def p_sample(self, model, x_t: torch.Tensor, k: int, s: torch.Tensor, variance_type = 'beta_tilde') -> torch.Tensor:
        """
        Sample from p(x_t^{k-1} | x_t^k, s_{t-1})
        h : (n_samples, reservoir_site) — ESN conditioning
        """
        k_tensor = torch.tensor([k], device=x_t.device).expand(x_t.shape[0])
        eps = model(x_t, k_tensor, h=s) 
        alpha_k     = self.scheduler.alpha[k]
        alpha_bar_k = self.scheduler.alpha_bar[k]
        beta_k      = self.scheduler.betas[k]
        alpha_bar_k_prev = self.scheduler.alpha_bar[k-1] # \bar{\alpha_{k-1}} 

        mu = (1 / torch.sqrt(alpha_k)) * (x_t - (beta_k / torch.sqrt(1 - alpha_bar_k)) * eps)

        if k > 0:
            z = torch.randn_like(x_t)
            if variance_type == 'beta_tilde':
                beta_tilde_k = (1-alpha_bar_k_prev)/(1-alpha_bar_k)*beta_k
                sigma_k = torch.sqrt(beta_tilde_k)
            elif variance_type == 'beta':
                sigma_k = torch.sqrt(beta_k)
            else:
                raise NameError(f"Select a valid variance type for the reverse process : ['beta_tilde','beta'], got {variance_type}")
            x_prev = mu + sigma_k * z
        else:
            x_prev = mu 

        return x_prev

    @torch.no_grad()
    def reverse_sampling(self, DeNetmodel, esnModel : ESN, y, c =None, n_samples=100, 
                         step_to_show = [75, 50, 25, 1], variance_type = 'beta_tilde') -> Tuple[torch.Tensor, list]:
        """
        DDPM Reverse sampler q(x_{t+1}|h_t)
        y : target series y_t (T,N)
        c : context c_t (T,M)
        """

        if c is not None:
            assert c.shape[0] == y.shape[0], "c and y should have the same time length"    
            c = c.float() 
            cond = torch.cat([y,c], dim=1)
        else:
            cond = y.clone()

        S = esnModel.get_states(cond) # s_1, ..., s_T
        last_s = S[-1].unsqueeze(0).expand(n_samples, -1) # (n_samples, reservoir_size)
        y_t = torch.randn(n_samples, y.shape[-1])

        show_list = []
        if self.scheduler.num_train_timesteps in step_to_show:
            show_list.append(y_t)

        for k in reversed(range(self.scheduler.num_train_timesteps)):
            y_t = self.p_sample(DeNetmodel, y_t, k=k, s=last_s, variance_type = variance_type)
            if k in step_to_show:
                show_list.append(y_t)

        return y_t, show_list
    
    @torch.no_grad()
    def sample_trajectory(self, DeNetmodel, esnModel: ESN, y, c=None, 
                          n_steps=250, n_samples=1, variance_type='beta_tilde') -> torch.Tensor:
        """ 
        Sample a trajectory. It only works with n_samples = 1 for now. 
        """
        if c is None:
            cond = y.float().clone()
        else: raise ValueError("The method is not yet implementing the case with additional covariates")

        # Get states s_1, ..., s_T, S shape (T, reservoir_size)
        S = esnModel.get_states(cond)   

        series = []
        for _ in tqdm(range(n_steps), desc="Trajectory", leave=False):

            # Last state as conditioning
            last_s = S[-1].unsqueeze(0).expand(n_samples, -1)  # (n_samples, reservoir_size)

            # Sample noise y ~ N(0,I), y shape (n_samples, N)
            y_t = torch.randn(n_samples, y.shape[-1])
            # Sample y_t ~ p(y_t | s_{t-1}) 
            for k in reversed(range(self.scheduler.num_train_timesteps)):
                y_t = self.p_sample(DeNetmodel, y_t, k=k, s=last_s, variance_type=variance_type)

            series.append(y_t.clone())  # (n_samples, N)

            # Get the 
            S = esnModel.update_states(y_t.squeeze(0).unsqueeze(-1))  # expects (N, 1)

        return torch.stack(series, dim=0)  # (n_steps, n_samples, N)
        

