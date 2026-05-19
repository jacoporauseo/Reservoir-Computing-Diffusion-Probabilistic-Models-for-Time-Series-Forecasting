from TimeGradUtils.diffModules import BaseScheduler, DDPM_TimeGrad
from TimeGradUtils.denoisingRNN import DenoisingNetTimeGrad, TimeGradRNN_FH, TrainSpec, TimeGradRNN_LH
import time 
import torch 
import torch.nn as nn 
import copy 
from tqdm import tqdm 
import torch.nn.functional as F


###########################################################################################################################
# Training TimeGrad Complete History
###########################################################################################################################


def training_TimeGrad_FH(
    y: torch.Tensor,
    c: torch.Tensor, 
    DeNetmodel: DenoisingNetTimeGrad, 
    RNNModel: TimeGradRNN_FH, 
    diffusion: DDPM_TimeGrad, 
    ts: TrainSpec
    ):
    """
    TIME GRAD ON FULL HISTORY. NO LOOKBACK PERIOD
    y : target tensor (T,N)
    c : additional time serires for conditioning (T,M)
    DeNetmodel: denoising entwork epsilon_theta(x^k_t,k,h_{t-1}) 
    RNNModel: RNN model 
    diffusion: TimeGrad diffusion module
    ts: data object with training and model specifications
    """
    start = time.time()

    T, N = y.shape
    y = y.float()
    y_pred = y[ts.t0:]   # y_t for predictions for t \in [T-t0,...,T]
    
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

    
    optimizer = torch.optim.Adam(
        list(DeNetmodel.parameters()) + list(RNNModel.parameters()), 
        lr=ts.lr
    )

    for epoch in tqdm(range(ts.epochs), leave=False, desc="Training"):

        DeNetmodel.train()
        RNNModel.train()

        # Full history pass: (T, hidden_size)
        h_history = RNNModel(cond.unsqueeze(0)).squeeze(0)

        # Shift by 1: h_{t0-1},...,h_{T-2} conditions y_{t0},...,y_{T-1}
        h_predWind = h_history[ts.t0-1 : T-1]   # (T-t0, hidden_size)

        # Sample random batch
        idx     = torch.randint(0, T - ts.t0, (ts.batch_size,))
        y_batch = y_pred[idx]        # (B, N)
        h_batch = h_predWind[idx]    # (B, hidden_size)

        # Sample diffusion steps
        k = torch.randint(0, diffusion.scheduler.num_train_timesteps, (ts.batch_size,))

        # Forward diffusion + denoising
        x_noisy, noise = diffusion.q_sample(y_batch, k=k)
        eps_pred       = DeNetmodel(x_noisy, k=k, h=h_batch)
        loss           = F.mse_loss(eps_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss        = loss.item()
            best_DeNet_state = copy.deepcopy(DeNetmodel.state_dict())
            best_RNN_state   = copy.deepcopy(RNNModel.state_dict())
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
    RNNModel.load_state_dict(best_RNN_state)

    end = time.time() 
    time_to_train = end - start 
    print("Time to train: ", time_to_train)
    return losses, best_loss, time_to_train

###########################################################################################################################
# Training TimeGrad Limited History
###########################################################################################################################


def get_chucks_x(x, Lookback, t0=None, augment=False):
    if t0 is None:
        t0 = Lookback   
    x_t    = torch.stack([x[i]               for i in range(t0, len(x))])         
    x_prev = torch.stack([x[(i-Lookback):i]  for i in range(t0, len(x))])         
    return x_t, x_prev

def training_TimeGrad_LH(
    y: torch.Tensor,
    DeNetmodel: DenoisingNetTimeGrad, 
    RNNModel: TimeGradRNN_LH, 
    diffusion: DDPM_TimeGrad, 
    ts: TrainSpec
    ):
    """ Train TimeGrad: joint training of RNN and DenoiseNetwork"""
    start = time.time()

    DeNetmodel.train()
    RNNModel.train()

    # Single optimizer for both networks
    optimizer = torch.optim.Adam(list(DeNetmodel.parameters()) + list(RNNModel.parameters()), lr=ts.lr)

    T, N = y.shape
    hidden_size = RNNModel.hidden_size

    # Precompute chunks — shape (T-t0, N) and (T-t0, L, N)
    y_t, y_prev = get_chucks_x(y, t0=ts.t0, Lookback=ts.Lookback)

    # Precompute ALL hidden states once before training: Each row i gives h for window ending at t0+i
    with torch.no_grad():
        h_all = RNNModel(y_prev)  

    losses = []
    best_loss        = float('inf')
    best_DeNet_state = None
    best_RNN_state   = None
    patience_counter = 0

    for epoch in tqdm(range(ts.epochs), leave=False, desc="Training"):

        # Sample random batch indices
        idx = torch.randint(0, T - ts.t0, (ts.batch_size,))

        x_batch  = y_t[idx]           
        xp_batch = y_prev[idx]        

        # Compute h history
        h_batch = RNNModel(xp_batch) 

        # Sample a random timestep
        k = torch.randint(0, diffusion.scheduler.num_train_timesteps, (ts.batch_size,))  

        # Forward diffusion
        x_noisy, noise   = diffusion.q_sample(x_batch, k=k)  

        # Noise prediction
        eps_pred         = DeNetmodel(x_noisy, k=k, h=h_batch) 
        loss             = F.mse_loss(eps_pred, noise)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        if loss.item() < best_loss:
            best_loss        = loss.item()
            best_DeNet_state = copy.deepcopy(DeNetmodel.state_dict())
            best_RNN_state   = copy.deepcopy(RNNModel.state_dict())
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

    # Restore best weights for both (!!!) networks
    DeNetmodel.load_state_dict(best_DeNet_state)
    RNNModel.load_state_dict(best_RNN_state)

    end = time.time()
    time_to_train = end - start
    print("The training took :", time_to_train)
    
    return losses, best_loss, time_to_train