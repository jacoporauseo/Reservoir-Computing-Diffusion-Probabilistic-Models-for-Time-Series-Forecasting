import torch 
from RwdnUtils.moduleRwdn import RWDenoisingNetwork
import time
import torch.nn.functional as F
from RwdnUtils.diffusionRwdn import DDPM_RWDN
from EsnUtils.ModuleESN import ESN

def training_RWDN(
        x       : torch.Tensor,        # (T, N) clean AR(1) data
        esn     : ESN,                 # ESN model
        model   : RWDenoisingNetwork,
        diffusion : DDPM_RWDN,
        K       : int,
        method  : str = 'Ridge',
        t0 : int = 200):

    start = time.time()
    T, N = x.shape

    # Define [x_{t0}, ..., x_T]
    x_t = x[t0:]

    assert K == diffusion.scheduler.num_train_timesteps == model.K, f"Inconsistent K: {K}, {diffusion.scheduler.num_train_timesteps}, {model.K}"
    assert N == model.N, f"Inconsistent N: {N} vs {model.N}"

    # Compute ESN states (T, S)
    H = esn.get_states(X=x)  
    H = H[t0-1:T-1]

    losses_before = []
    losses_after  = []

    for k in range(K):
        k_tensor = torch.full((T-t0,), fill_value=k, dtype=torch.long)

        # Noised input + noise target
        x_k, epsilon = diffusion.q_sample(x_0=x_t, k=k_tensor) 

        # Check loss before fitting
        # I should use the model class, but for debugging i did this for now...
        U       = torch.cat([x_k, H], dim=1)                          
        Z       = model.sigma(U @ model.A[k].T + model.b[k].squeeze(1))  
        eps_hat = Z @ model.W[k]                                       
        loss_bt = F.mse_loss(eps_hat, epsilon)
        losses_before.append(loss_bt.item())

        # Estimate obs map
        model.fit(X_k=x_k, H_t=H, E_k=epsilon, k=k, method=method)

        #Check Loss after fitting
        eps_hat2 = Z @ model.W[k]   # use the forward... here just for debug was easy to reycle Z since it is the same.
        loss_at  = F.mse_loss(eps_hat2, epsilon)
        losses_after.append(loss_at.item())

        if k % 25 == 0:
            print(f"k={k:3d} | Loss before: {loss_bt.item():.4f} | Loss after: {loss_at.item():.6f}")

    elapsed = time.time() - start
    print(f"\nTraining complete :=) ! Time: {elapsed:.2f}s")
    return losses_before, losses_after, elapsed