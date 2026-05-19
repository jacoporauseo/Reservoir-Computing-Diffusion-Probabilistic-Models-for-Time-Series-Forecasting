import torch 
import torch.nn.functional as F
from EsnUtils.ModuleESN import ESN, DenoisingNetTimeGrad
from TimeGradUtils.denoisingRNN import TrainSpec
from EsnUtils.diffusionESN import ESN_DDPM
import time
from tqdm import tqdm 
import copy  


def training_ESN_Score(
    y: torch.Tensor,
    c: torch.Tensor, 
    DeNetmodel: DenoisingNetTimeGrad, 
    EsnModel: ESN, 
    diffusion: ESN_DDPM, 
    ts: TrainSpec
    ):
    """
    EsnGrad training according to Algorithm 5

    Parameters
    ---------
    y : target tensor (T,N)
    c : additional time serires for conditioning (T,M)
    DeNetmodel: denoising entwork epsilon_theta(x^k_t,k,s_{t-1}) 
    EsnModel: ESN model 
    diffusion: TimeGrad diffusion module
    ts: data object with training and model specifications
    """
    start = time.time()
    T, N = y.shape
    y = y.float()
    y_pred = y[ts.t0:]   # y_t for predictions for t \in [t0,...,T]
    
    if c is not None:
        assert c.shape[0] == y.shape[0], "c and y should have the same time length"    
        c = c.float() 
        cond = torch.cat([y,c], dim=1)
    else:
        cond = y.clone()

    losses = []
    best_loss        = float('inf')
    best_DeNet_state = None
    best_RNN_state   = None
    patience_counter = 0

    # ESN history: no need to recompute it every time since parameters are fixed
    S = EsnModel.get_states(cond)
    # Shift by 1: h_{t0-1},...,h_{T-2} conditions y_{t0},...,y_{T-1}
    h_predWind = S[ts.t0-1 : T-1]   # (T-t0, reservoir-size, 1)

    optimizer = torch.optim.Adam(DeNetmodel.parameters(), lr=ts.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=50, factor=0.5, min_lr=1e-6)

    for epoch in tqdm(range(ts.epochs), leave=False, desc="Training"):

        DeNetmodel.train()

        # Sample random batch
        idx     = torch.randint(0, T - ts.t0, (ts.batch_size,))
        y_batch = y_pred[idx]        # (B, N)
        s_batch = h_predWind[idx]    # (B, Reservoir_size)

        # Sample diffusion steps
        k = torch.randint(0, diffusion.scheduler.num_train_timesteps, (ts.batch_size,))

        # Forward diffusion + denoising
        x_noisy, noise = diffusion.q_sample(y_batch, k=k)
        eps_pred       = DeNetmodel(x_noisy, k=k, h=s_batch)
        loss           = F.mse_loss(eps_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step(loss.item())

        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss        = loss.item()
            best_DeNet_state = copy.deepcopy(DeNetmodel.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if epoch % 50 == 0 and ts.print_loss:
            print(f"Epoch {epoch}/{ts.epochs} | Loss: {loss.item():.4f} | Patience: {patience_counter}/{ts.patience}")

        if patience_counter >= ts.patience:
            if ts.print_loss:
                print(f"Early stopping at epoch {epoch} | Best loss: {best_loss:.4f}")
            break

        if loss.item() <= ts.min_loss: 
            break 

    DeNetmodel.load_state_dict(best_DeNet_state)
    end = time.time()
    time_len = end - start
    print("Training time ", time_len)

    return losses, time_len