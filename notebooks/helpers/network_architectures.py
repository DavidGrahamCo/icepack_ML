import torch
import torch.nn as nn

class switch_shape(nn.Module):
    # (batch, time, channels) to (batch, channels, time) which is what conv1d expects
    def forward(self,x):
        return x.transpose(1,2)
    
def Linear(timesteps, channels, unknowns=1):
    # defines a basic linear regression, a baseline.
    return nn.Sequential(nn.Flatten()
                        ,nn.Linear(timesteps*channels,unknowns)
                        )

def MLP_1hidden(timesteps, channels, middle_layer = 64, unknowns=1):
    # defines a basic Multi-Layer Perceptron Neural Net with 1 hidden layer and middle_layer as middle increment. unknowns is the number of outputs
    return nn.Sequential(nn.Flatten()
                        ,nn.Linear(timesteps*channels, middle_layer)
                         ,nn.ReLU()
                         ,nn.Linear(middle_layer,unknowns)
                        )

def CNN_1(timesteps, channels, kernel_size = 5, out_num=64, pool_num=3, unknowns=1):
    return nn.Sequential(switch_shape()
                        ,nn.Conv1d(channels, out_num, kernel_size, padding = kernel_size // 2)
                        ,nn.ReLU()
                        ,nn.AvgPool1d(pool_num)
                        ,nn.Flatten()
                        ,nn.Linear( out_num * (timesteps//pool_num),unknowns)
                )



def CNN_complex(timesteps, channels, conv_channels = 64, kernel_size = 5, conv_per_block=2, n_blocks = 3, pool_num = 2, linear_layer = 32, unknowns = 1):
    layers = []
    layers.append(switch_shape())
    c_in = channels
    for b in range(n_blocks):
        for i in range(conv_per_block):
            layers.append(nn.Conv1d(c_in, conv_channels, kernel_size, padding = kernel_size//2))
            layers.append(nn.ReLU())
            c_in = conv_channels
        layers.append(nn.AvgPool1d(pool_num))
    layers.append(nn.AdaptiveAvgPool1d(1))
    layers.append(nn.Flatten())
    layers.append(nn.Linear(c_in,linear_layer))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(linear_layer,unknowns))
    return nn.Sequential(*layers)


MODEL_CHOICES = {"MLP_1":MLP_1hidden, "CNN_1": CNN_1, "CNN":CNN_complex, "Linear":Linear}

def blank_model(choice, timesteps, channels, **optional):
    return MODEL_CHOICES[choice](timesteps, channels, **optional)






class GaussianLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.nll = nn.GaussianNLLLoss()

    def forward(self, out, y):
        mu = out[:, :1]#match y
        log_sigma = torch.clamp(out[:, 1:], -10.0, 5.0)  #clamp
        
        return self.nll(mu, y, torch.exp(2 * log_sigma)) 
    
def single_pred(out):
    #For MSE, just makes the functions match
    return out, None

def gauss_pred(out):
    # (mu,log(sigma)) to (mu, sigma)
    return out[:,:1], torch.exp(torch.clamp(out[:, 1:], -10.0, 5.0))
    
LOSS_CHOICES = {
    "MSE":{"loss":nn.MSELoss(), "unknowns":1, "eval_method":single_pred},
    "GAUSS":{"loss":GaussianLoss(), "unknowns":2, "eval_method":gauss_pred}
}