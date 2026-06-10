import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from scipy.stats import t as student_t
from typing import Tuple, List
from scipy import stats
import torch

class AR1(ABC):   
    @abstractmethod
    def generate_trajectory(self, T: int) -> torch.Tensor:
        "Create a trajectory from the AR process"
        pass
    
    @abstractmethod
    def conditional_pdf(self, x: np.ndarray, c:float):
        """Conditional pdf given x_{t-1}"""
        pass
    def sample_prev(self,T:int): 
        """Return an array of shape [T,2] with [x_t,x_{t-1}]"""
        x = self.generate_trajectory(T = T + 1)
        x = torch.cat([x[1:], x[:-1]], dim=1)
        return x 


class AR_normal(AR1):
    """Class for AR1 process using Normal innovations"""
    def __init__(self, phi=0.8, sigma=1, seed = 123):
        self.phi = phi
        self.sigma = sigma  # standard deviation

    def generate_trajectory(self, T:int=1000):
        x = np.zeros(T)
        x[0] = np.random.normal(0, self.sigma)
        for t in range(1, T):
            x[t] = self.phi * x[t-1] + np.random.normal(0, self.sigma)
        return torch.from_numpy(x.reshape(-1, 1)).float()

    def conditional_pdf(self, x: np.ndarray, c: float):
        mu = self.phi * c                          
        p = (1 / (np.sqrt(2 * np.pi) * self.sigma) * np.exp(-(x - mu)**2 / (2 * self.sigma**2)))  
        return p
       

class AR_studentt(AR1):
    """Class for AR1 process using student t innovations"""
    def __init__(self, phi=0.8, scale=1, df=5, seed=123):
        self.phi = phi
        self.scale = scale
        self.df = df
        self.seed = seed

    def generate_trajectory(self, T: int = 100000):
        np.random.seed(self.seed)
        x = np.zeros(T)
        x[0] = np.random.standard_t(self.df) * self.scale
        for t in range(1, T):
            x[t] = self.phi * x[t-1] + np.random.standard_t(self.df) * self.scale
        return torch.from_numpy(x.reshape(-1, 1)).float()

    def conditional_pdf(self, x: np.ndarray, c: float):
        mu = self.phi * c
        x_grid = np.linspace(-4, 4, 100)
        p = student_t.pdf(x_grid, df=self.df, loc=mu, scale=self.scale)
        return x_grid, p


class MaxAr(AR1):
    def __init__(self, phi, sigma = 1):
        self.phi = phi 
        self.sigma = 1

    def generate_trajectory(self, T: int):
        "Create a trajectory from the AR process"
        y = np.zeros(T+1)
        y[1:] = np.max(y[:T]*self.phi,0) + np.random.normal(0,self.sigma,T)
        return torch.from_numpy(y[1:].reshape(-1, 1)).float()

    def conditional_pdf(self, x: np.ndarray, c: float):
        """
        p(x_t = x | x_{t-1} = c)
        x : grid of values to evaluate
        c : conditioning value x_{t-1}
        """
        mean = self.phi * np.maximum(c, 0)
        return stats.norm.pdf(x, loc=mean, scale=self.sigma)


class SQRTAr(AR1):
    def __init__(self, phi=0.6, sigma=1.0):
        self.phi = phi
        self.sigma = sigma

    def generate_trajectory(self, T):
        y = np.zeros(T+1)
        y[0] = np.random.uniform(1, 4) 
        for t in range(T):
            y[t+1] = self.phi * np.sqrt(np.abs(y[t])) + np.random.normal(0, self.sigma)
        return torch.from_numpy(y[1:].reshape(-1, 1)).float()

    def conditional_pdf(self, x, c):
        loc = self.phi * np.sqrt(abs(c))
        return stats.norm.pdf(x, loc=loc, scale=self.sigma)


class ARCH1(AR1):
    def __init__(self, alpha=0.6):
        self.alpha = alpha   

    def generate_trajectory(self, T):
        y = np.zeros(T+1)
        y[0] = np.random.normal(0, 1)
        for t in range(T):
            sigma_t = np.sqrt(0.1 + self.alpha * y[t]**2)
            y[t+1] = sigma_t * np.random.normal(0, 1)
        return torch.from_numpy(y[1:].reshape(-1, 1)).float()

    def conditional_pdf(self, x, c):
        sigma = np.sqrt(0.1 + self.alpha * c**2)
        return stats.norm.pdf(x, loc=0, scale=sigma)


from scipy import stats

class GARCH11(AR1):
    def __init__(self, omega=0.1, alpha=0.1, beta=0.8, sigma=None):
        """
        x_t = sigma_t * eta_t,  eta_t ~ N(0,1)
        sigma_t^2 = omega + alpha*x_{t-1}^2 + beta*sigma_{t-1}^2
        """
        self.omega = omega
        self.alpha = alpha
        self.beta  = beta
        assert alpha + beta < 1, "Need alpha+beta < 1 for stationarity"
        self.v = sigma if sigma is not None else omega / (1 - alpha - beta)

    def generate_trajectory(self, T, seed = 123):
        torch.manual_seed(seed)
        np.random.seed(seed)
        y      = np.zeros(T+1)
        sigma2 = np.zeros(T+1)
        sigma2[0] = self.v
        y[0]      = np.random.normal(0, np.sqrt(sigma2[0]))

        for t in range(T):
            sigma2[t+1] = self.omega + self.alpha * y[t]**2 + self.beta * sigma2[t]
            y[t+1]      = np.sqrt(sigma2[t+1]) * np.random.normal(0, 1)  

        next_sigma = self.omega + self.alpha * y[-1]**2 + self.beta * sigma2[-1]

        y      = torch.from_numpy(y[1:].reshape(-1, 1)).float()
        sigma2 = torch.from_numpy(sigma2[1:].reshape(-1, 1)).float()

        return y, sigma2, next_sigma

    def conditional_pdf(self, x, sigma2_t):
        """
        p(x_t | F_{t-1}) = N(0, sigma2_t)
        """
        return stats.norm.pdf(x, loc=0, scale=np.sqrt(sigma2_t)) 

    def squared_too(self, T):
        y, _, _ = self.generate_trajectory(T)
        x = y**2
        return torch.concat([y, x], dim=1)

    def trajectory_squares(self, T):
        y, _, last_sigma = self.generate_trajectory(T)
        return y**2, last_sigma