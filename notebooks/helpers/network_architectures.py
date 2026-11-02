import torch
import torch.nn as nn
from pathlib import Path
import json
import numpy as np

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

def rebuild_net(train_config, EXPERIMENT_DIR,device):
    model = blank_model(train_config["model_choice"],train_config["input_window"], len(train_config["training_vars"]), **train_config["model_options"] )
    state_dict = torch.load(EXPERIMENT_DIR /"model.pth", map_location = device)
    model.load_state_dict(state_dict)
    return(model)

def warm_start_mse(cold_model, MSE_DIR, device):
    #takes in the CNN with 2 outputs and matches the weights and biases to the pretrained MSE model. Uses 0 as the starting weights and rmse on validation set as the started bias for new log(sigma) output
    #returns the initialized model and the mse net train_config
    mdir = Path(MSE_DIR)
    with open(mdir / 'train_config.json') as f:
        train_config = json.load(f)
    warm_state = torch.load(mdir /"model.pth", map_location = device)    
    #print(train_config)
    starting_sigma = np.sqrt(train_config["best_error"]) #rmse is standard deviation when bias = 0, might count as data leakage?
    
    cold_state = cold_model.state_dict()
    last_weight = list(cold_state)[-2]
    last_bias = list(cold_state)[-1]
    

    for key, val in warm_state.items(): #key is layer.weight or layer.bias, val is the tensor weights (or biases)
        if key == last_weight:
            #print(cold_state[key].shape)
            cold_state[key][0] = warm_state[key][0] #mu stays the same as mse output
            cold_state[key][1] = 0 #start with no info on log(sigma)
            continue
        if key == last_bias:
            cold_state[key][0] = warm_state[key][0]
            cold_state[key][1] = np.log(starting_sigma) #all sds start uniformly at rmse of model
            continue
        assert cold_state[key].shape == val.shape
        cold_state[key] = val
    
    cold_model.load_state_dict(cold_state)
    
    
    return cold_model, train_config
    
    

MODEL_CHOICES = {"MLP_1":MLP_1hidden, "CNN_1": CNN_1, "CNN":CNN_complex, "Linear":Linear}


def blank_model(choice, timesteps, channels, **optional):
    return MODEL_CHOICES[choice](timesteps, channels, **optional)






class GaussianLoss(nn.Module):
    def __init__(self, min_log_sigma = -10.0, max_log_sigma = 5.0):
        super().__init__()
        self.low = min_log_sigma
        self.high = max_log_sigma
        self.nll = nn.GaussianNLLLoss()

    def forward(self, out, y):
        mu = out[:, :1]#match y
        log_sigma = torch.clamp(out[:, 1:], self.low , self.high)  #clamp
        
        return self.nll(mu, y, torch.exp(2 * log_sigma)) 
    
def gauss_pred(low = -10.0, high = 5.0):
    # (mu,log(sigma)) to (mu, sigma)
    return lambda out: (out[:,:1], torch.exp(torch.clamp(out[:, 1:], low, high)))

    
    
def single_pred(out):
    #For MSE, just makes the functions match
    return out, None



def probability_pred(out):
    # (z) to (p = 1/(1+exp(-z)))
    return torch.sigmoid(out), None
    
def gauss_choice(low = -10, high = 5):
    assert (low < high)
    return {"loss":GaussianLoss(low, high), "unknowns":2, "eval_method":gauss_pred(low,high)}

LOSS_CHOICES = {
    "MSE":{"loss":nn.MSELoss(), "unknowns":1, "eval_method":single_pred},
    "BCE":{"loss":nn.BCEWithLogitsLoss(), "unknowns":1, "eval_method":probability_pred}
}

LOSS_CHOICES["GAUSS"] = gauss_choice(low = -10, high = 5)