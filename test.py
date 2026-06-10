from TimeGradUtils.epsilon_theta import EpsilonTheta 
from TimeGradUtils.denoisingRNN import TimeGradRNN_FH 
import torch 


N = 2
T = 100
x = torch.randn(size = (T,1,N), dtype=torch.float32)
rnn = TimeGradRNN_FH(input_size=N, hidden_size=16, cell = 'LSTM')
s = rnn(x)

print("Shape: \n")
print(s.shape)
print("Tensor: \n")
print(s)

k = torch.randint(low = 0, high=100,size = (T,))

epsilon = EpsilonTheta(target_dim=N,cond_length=16) 

e = epsilon.forward(inputs = x,time = k, cond=s)

print(e.shape) # (batch, 1, N)
