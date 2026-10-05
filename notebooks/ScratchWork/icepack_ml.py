# icepack_ml.py
import json
import numpy as np
import torch

def make_index(run_ids, T, input_window):
    # build the indicies for the input series, have step incase we want over
    index = []
    for r in run_ids:
        t0 = 0
        while t0 + input_window <= T:
            index.append([int(r), t0])
            t0 += input_window
    return index


class IcepackDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, index, window, mu, sd):
        self.X = X# (N, T, C)
        self.Y = Y# (N, P)
        self.index = index #array of [run, start]
        self.window = window
        self.mu, self.sd = mu,sd  

    def __getitem__(self, ind):
        run,start = self.index[ind]
        xnorm = self.X[run, start:start + self.window]# (Window, C)
        return (xnorm - self.mu)/self.sd, self.Y[run]

    def __len__(self):
        return len(self.index)
