import torch
import torch.nn as nn
import numpy as np 
from typing import Tuple, List
from tqdm import tqdm 
from abc import ABC, abstractmethod

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
            betas = (
                torch.linspace(beta_1**0.5, beta_K**0.5, num_train_timesteps) ** 2
            )
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


class Diffusion(ABC):
    """Class to be completed"""

    @abstractmethod
    def q_sample(self, x_0: torch.Tensor, k: int, noise = None):
        pass


class DDPM:
    """The idea for the class is taken from KAIST Diffusion Model Course"""
    def __init__(self, scheduler : BaseScheduler):
        self.scheduler = scheduler

    def q_sample(self, x_0 : torch.Tensor, k: int, noise = None):
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
    def p_sample(self, model, x_t: torch.Tensor, k:int, y : torch.Tensor):
        r"""
        Sample from the reverse  p(x_{k-1} | x_t) ~ N(mu,sigma)
        """
        t_tensor = torch.tensor([k], device=x_t.device).expand(x_t.shape[0])
        eps = model(x_t, t_tensor, y=y)
        
        alpha_k     = self.scheduler.alpha[k]
        alpha_bar_k = self.scheduler.alpha_bar[k]
        beta_k      = self.scheduler.betas[k]
        alpha_bar_k_prev = self.scheduler.betas[k-1] # \bar{\alpha_{k-1}} 

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

    
class DDIM(DDPM):
    """DDIM Reverse process"""

    @torch.no_grad()
    def p_sample_ddim(self, model, x_t, k: int, t_prev: int, y=None, eta: float = 0.0):
        t_tensor = torch.tensor([k], device=x_t.device).expand(x_t.shape[0])  

        eps = model(x_t, t_tensor, y=y)

        alpha_bar_k = self.scheduler.alpha_bar[k]
        
        alpha_bar_k_prev = self.scheduler.alpha_bar[t_prev] if t_prev >= 0 else torch.tensor(1.0, device=x_t.device)
        beta_k = 1 - alpha_bar_k / alpha_bar_k_prev

        x0_pred = (x_t - torch.sqrt(1 - alpha_bar_k) * eps) / torch.sqrt(alpha_bar_k)

        sigma_t = eta * torch.sqrt((1 - alpha_bar_k_prev) / (1 - alpha_bar_k) * beta_k)
        
        dir_xt_var = torch.clamp(1 - alpha_bar_k_prev - sigma_t**2, min=0.0)
        dir_xt = torch.sqrt(dir_xt_var) * eps

        x_prev = torch.sqrt(alpha_bar_k_prev) * x0_pred + dir_xt + sigma_t * torch.randn_like(x_t)

        return x_prev

    @torch.no_grad()
    def reverse_sampling_ddim(self, model, n_samples: int, y: float, n_steps: int = 50):
        """
        Steps governs the number of DDIM steps to use.
        """
        x = torch.randn((n_samples, 1))
  
        y_tensor = torch.full((n_samples, 1), y)

        timesteps = torch.linspace(0, self.scheduler.num_train_timesteps - 1, n_steps).long()
        timesteps = timesteps.flip(0) 

        for i, k in enumerate(timesteps):
            t_prev = int(timesteps[i + 1].item()) if i + 1 < len(timesteps) else -1
            t_val = int(k.item())  
            x = self.p_sample_ddim(model, x, t_val, t_prev, y=y_tensor)

        return x


class DDPM_TimeGrad(DDPM):
    """DDPM with RNN conditioning for TimeGrad"""
    @torch.no_grad()
    def p_ddpm(self, eModel, x_t: torch.Tensor, k: int, h: torch.Tensor, variance_type = 'beta_tilde'):
        """
        Sample from p(x^{k-1} | x^k, h)
        h : (n_samples, hidden_size) — RNN conditioning
        """
        k_tensor = torch.tensor([k], device=x_t.device).expand(x_t.shape[0])
        eps = eModel(x_t, k_tensor, h=h) 
        # Clip predicted noise — prevents runaway denoising steps
        # eps = torch.clamp(eps, -3.0, 3.0)
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
            else:
                sigma_k = torch.sqrt(beta_k)
            x_prev = mu + sigma_k * z
        else:
            x_prev = mu 

        return x_prev
    
    @torch.no_grad()
    def p_ddim(self, eModel, x_t: torch.Tensor, k: int, h: torch.Tensor, eta: float):
        """DDIM variational distribution"""
        k_tensor = torch.tensor([k], device=x_t.device).expand(x_t.shape[0])
        eps = eModel(x_t, k_tensor, h=h)

        alpha_bar_k      = self.scheduler.alpha_bar[k]
        alpha_bar_k_prev = self.scheduler.alpha_bar[k] if k > 0 else torch.tensor(1.0, device=x_t.device)
        beta_k           = self.scheduler.betas[k]

        x_0t  = (x_t - torch.sqrt(1 - alpha_bar_k) * eps) / torch.sqrt(alpha_bar_k)
        sigma = eta * torch.sqrt((1 - alpha_bar_k_prev) / (1 - alpha_bar_k) * beta_k)

        dir_coef = torch.sqrt(torch.clamp(1 - alpha_bar_k_prev - sigma**2, min=0.0))
        dir_xt   = dir_coef * (x_t - torch.sqrt(alpha_bar_k) * x_0t) / torch.sqrt(1 - alpha_bar_k)

        x_prev = torch.sqrt(alpha_bar_k_prev) * x_0t + dir_xt
        if k > 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)

        return x_prev
    
    @torch.no_grad()
    def reverse_sampling_ddim(self, DeNetmodel, RNNModel, y, c = None, n_samples=100, step_to_show = [75, 50, 25, 1], eta = 0.0,num_inference_steps=50):
        """
        NOT WORKING. One should adjust the the indexing to allow for skipping steps in the reverse
        """

        # Build subsampled timestep sequence
        T = self.scheduler.num_train_timesteps
        step_ratio = T // num_inference_steps
        timesteps = list(reversed(range(0, T, step_ratio)))  

        if c is not None:
            cond = torch.cat([y,c], dim = 1)
        else:
            cond = y.clone()

        current_context = cond.unsqueeze(0).expand(n_samples, -1, -1).clone()

        h_out = RNNModel(current_context)
        h = h_out[:, -1, :] if h_out.ndim == 3 else h_out 


        y_t = torch.randn(n_samples, y.shape[-1])

        show_list = []
        if self.scheduler.num_train_timesteps in step_to_show:
            show_list.append(y_t)

        for k in timesteps:
                # Need alpha_bar_prev at the *previous timestep in subsequence*, not k-1
                y_t = self.p_ddim(DeNetmodel, y_t, k=k, h=h, eta=eta)

        return y_t, show_list

    @torch.no_grad()
    def reverse_sampling(self, DeNetmodel, RNNModel, y, c = None, n_samples=100, step_to_show = [75, 50, 25, 1], variance_type = 'beta_tilde'):
        """
        DDPM Reverse sampler q(x_{t+1}|h_t)
        y : target series (T,N)
        c : context (T,M)
        """
        if c is not None:
            cond = torch.cat([y,c], dim = 1)
        else:
            cond = y.clone()

        current_context = cond.unsqueeze(0).expand(n_samples, -1, -1).clone()

        h_out = RNNModel(current_context)
        h = h_out[:, -1, :] if h_out.ndim == 3 else h_out 


        y_t = torch.randn(n_samples, y.shape[-1])
        show_list = []
        if self.scheduler.num_train_timesteps in step_to_show:
            show_list.append(y_t)

        for k in reversed(range(self.scheduler.num_train_timesteps)):
            y_t = self.p_ddpm(DeNetmodel, y_t, k=k, h=h, variance_type = variance_type)
            if k in step_to_show:
                show_list.append(y_t)

        return y_t, show_list

    @torch.no_grad()
    def sample_trajectory(self, DeNetmodel: nn.Module, RNNModel: nn.Module, y: torch.Tensor, c = None, 
                          n_steps: int = 500, n_samples: int = 1, variance_type='beta_tilde'):
           
        if c is None: 
            c = y.clone() 
        else:
            raise NotImplementedError("The case with c not none is not yet implemented in this method")

        current_context = c.unsqueeze(0).expand(n_samples, -1, -1).clone()  # (n_samples, T, N)
        generated = []

        for _ in tqdm(range(n_steps), desc="Generation", leave=False):

            x_t, _ = self.reverse_sampling(DeNetmodel, RNNModel, y=current_context, c=None, 
                                            n_samples=n_samples, variance_type=variance_type)
            generated.append(x_t)

            # Slide context window forward
            current_context = torch.cat([current_context[:, 1:, :], x_t.unsqueeze(1)], dim=1)  # (n_samples, T, N)

        return torch.stack(generated, dim=1)  # (n_samples, n_steps, N) 