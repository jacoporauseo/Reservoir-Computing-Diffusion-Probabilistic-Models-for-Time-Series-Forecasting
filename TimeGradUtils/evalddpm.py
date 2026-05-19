from TimeGradUtils.ArProcess import AR1 
from scipy.stats import wasserstein_distance
import numpy as np 
import torch 
import matplotlib.pyplot as plt 
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
# import ot


def plot_ar1_autocorr(y_ddpm : np.ndarray, true_phi : float, max_lags : int = 13, ):
    """Plot ACF and PACF of the y_ddpm series against true AR(1) autoregressive structure"""
    # Theoretical ACF and PACF
    lags = np.arange(0,max_lags)
    theoretical_acf = true_phi ** lags
    theoretical_pacf = np.zeros(len(lags))
    theoretical_pacf[0] = 1.0  
    theoretical_pacf[1] = true_phi  

    fig, ax = plt.subplots(1, 2, figsize=(10, 5), sharey=True)

    plot_acf(y_ddpm, lags=12, ax=ax[0], label='TimeGrad', color='blue', alpha=0.6)
    ax[0].vlines(lags, 0, theoretical_acf, color='orange', linewidth=1.5, label='Theoretical')
    ax[0].scatter(lags, theoretical_acf, color='orange', zorder=3)
    ax[0].set_title("ACF: TimeGrad vs Theoretical")
    ax[0].legend()

    # --- PACF ---
    plot_pacf(y_ddpm, lags=12, ax=ax[1], label='TimeGrad', color='blue', alpha=0.6)
    ax[1].vlines(lags, 0, theoretical_pacf, color='orange', linewidth=1.5, label='Theoretical')
    ax[1].scatter(lags, theoretical_pacf, color='orange', zorder=3)
    ax[1].set_title("PACF: TimeGrad vs Theoretical")
    ax[1].legend()
    
    plt.suptitle(f"ACF and PACF: TimeGrad vs Theoretical AR(1) with phi = {true_phi}")
    plt.ylim([min(theoretical_acf)-0.15, 1])
    plt.tight_layout()
    plt.show()

# def generate_with_residuals(ar : AR1, T: int):
#     """Return AR(1) series and residuals"""
#     x = ar.generate_trajectory(T=T).flatten()
#     residuals = x[1:] - ar.phi * x[:-1]  
#     return x.reshape(-1, 1), residuals.reshape(-1, 1)

# def residuals_ar1(x : np.ndarray, phi : float):
#     """get the residuals of an AR(1) model using the true phi"""
#     x = x.flatten()
#     residuals = x[1:] - phi * x[:-1]
#     return x, residuals.reshape(-1, 1)

# def compare_residuals_wasserstein(e1 : np.ndarray,e2 :np.ndarray):
#     """Compare residuals"""
#     e1 = e1.flatten()
#     e2 = e2.flatten()
#     return wasserstein_distance(e1, e2)

# def get_chunks(x, k):
#     """For an array of shape [T,] returns an array of shape [T-k,k] with the kth adiacent ts obs"""
#     return np.array([x[i:i+k+1] for i in range(len(x) - k -1)])

# def wasserstein_chunks(y1, y2, k : int):
#     """W_1 btw the AR(1) ground truth and DDPM traj"""
#     y_chunk_1 = get_chunks(y1, k)
#     y_chunk_2 = get_chunks(y2, k)
#     assert y_chunk_1.shape == y_chunk_2.shape
#     M = ot.dist(y_chunk_1, y_chunk_2, metric='euclidean')
#     n = y_chunk_1.shape[0]
#     a, b = np.ones((n,)) / n, np.ones((n,)) / n  
#     W = ot.emd2(a, b, M) 
#     G0 = ot.solve(M, a, b).plan
#     return W

def build_chunks(series: np.ndarray, lookback):
    """Returns chunks (T-k-1, k) and next values (T-k-1,)"""
    T = len(series)
    chunks    = np.array([series[t : t+lookback, 0]   for t in range(T - lookback)])
    nextvals  = np.array([series[t + lookback, 0]      for t in range(T - lookback)])
    return chunks, nextvals


def compute_conditional_wasserstein(
    x: np.ndarray,          # (T, 1) true DGP series
    y: np.ndarray,          # (T, 1) generated series
    lookback: int,          
    M: int = 200,           # number of chunks to sample
    knn: int = 50,          # number of neighbors
    ) -> dict:
    """
    Estimate E_{B~pi}[W1(mu_x(B), mu_y(B))] via kNN conditional density.
    """

    x_chunks, x_next = build_chunks(x, lookback)   
    y_chunks, y_next = build_chunks(y, lookback)   

    # Standardization needed only for he chunks not for the next value
    mean = x_chunks.mean(axis=0, keepdims=True) 
    std  = x_chunks.std(axis=0, keepdims=True) + 1e-8
    x_chunks_std = (x_chunks - mean) / std
    y_chunks_std = (y_chunks - mean) / std      

    # Sample M chunks at random 
    idx_queries = np.random.choice(len(x_chunks), size=M, replace=False)

    W1_list = []

    for idx in idx_queries:
        query = x_chunks_std[idx]            

        # Euclidean distance in np for the true
        dists_x = np.linalg.norm(x_chunks_std - query, axis=1)  
        nn_x    = np.argsort(dists_x)[1:knn+1]  
        mu_x    = x_next[nn_x] 

        # kNN in y 
        dists_y = np.linalg.norm(y_chunks_std - query, axis=1)  
        nn_y    = np.argsort(dists_y)[:knn]
        mu_y    = y_next[nn_y]       

        W1_list.append(wasserstein_distance(mu_x, mu_y))

    W1_list = np.array(W1_list)

    res = {
        "W1_mean":   W1_list.mean(),
        "W1_std":    W1_list.std(),
        "W1_all":    W1_list
        }

    return res