# This is a record of functions that multiple notebooks call
import torch


def make_index(run_ids, T, input_window):
    # takes the full run length (T) and run number indicies (run_ids) as well as the length of the training window.
    # Returns the run and timestep number in a list for each __ length input window.
    index = []
    for r in run_ids:
        t0 = 0
        while t0 + input_window <= T:
            index.append([int(r), t0])
            t0 += input_window
    return index




def read_mod_file(modfile):
    # Takes in a modfile, and returns a dictionary with each modified value and the value it was modified to
    # There is a library to do this, if this ever needs an update just switch to f90nml
    def parse(val):
        # Goes from fortran ids to python
        val = val.strip().rstrip(",")
        low = val.lower()
        if low in (".true.", ".false."):
            return low == ".true."
        try:
            return float(low.replace("d", "e"))
        except ValueError:
            return val.strip("'\"")          # strings stay strings

    param_dict = {}
    for line in modfile.read_text().splitlines():
        line = line.split("!")[0].strip()
        if line == '' or "=" not in line:
            continue
        k, v = line.split("=", 1)
        param_dict[k.strip()] = parse(v)
    return param_dict


def check_nans(ds, default=-9999):
    # check for any nans, infs, zeros or unset values in a dataset (Xarray)
    print(f"{'variable':12s} {'nan':>10s} {'default':>10s} {'inf':>10s} {'zero':>10s} {'total':>12s}")
    for name, da in ds.data_vars.items():
        a = da.values
        n_nan  = int(np.isnan(a).sum())
        n_sent = int(np.isclose(a, default).sum())
        n_inf  = int(np.isinf(a).sum())
        n_zero = int((a == 0).sum())
        flag = "  <--" if (n_nan or n_sent or n_inf) else ""
        print(f"{name:12s} {n_nan:>10d} {n_sent:>10d} {n_inf:>10d} {n_zero:>10d} {a.size:>12d}{flag}")
        


class IcepackDataset(torch.utils.data.Dataset):
    # Creates the IcepackDataset class, needed for pytorch
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