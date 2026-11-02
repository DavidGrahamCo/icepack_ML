# This is a record of functions that multiple notebooks call
import torch

def train(model, dataloader, optimizer, y_mu, y_sd, loss_fn):
    # one training pass through the loader
    model.train() #sets to training mode
    device = next(model.parameters()).device
    loss_tot = 0
    n=0
    for xbatch, ybatch in dataloader:
        ybatch_std = ((ybatch-y_mu)/y_sd).to(device)
        xbatch = xbatch.to(device)
        loss = loss_fn(model(xbatch), ybatch_std)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_tot += loss.item() * len(xbatch)
        n += len(xbatch)
    return loss_tot/n 
    
@torch.no_grad() # don't train on benchmark data
def validate(model, dataloader, y_mu, y_sd, loss_fn):
    model.eval() #sets to evaluate mode
    device = next(model.parameters()).device
    loss_tot = 0
    n=0
    for xbatch, ybatch in dataloader:
        ybatch_std = ((ybatch-y_mu)/y_sd).to(device)
        xbatch = xbatch.to(device)
        loss = loss_fn(model(xbatch), ybatch_std)
        loss_tot += loss.item() * len(xbatch)
        n += len(xbatch)
    return loss_tot/n

@torch.no_grad()
def evaluate(model, dataset, y_mu, y_sd, eval_method):
    model.eval()
    device = next(model.parameters()).device
    x_in = torch.stack([dataset[i][0] for i in range(len(dataset))]).to(device)
    mu,sigma = eval_method(model(x_in))
    mu = mu.cpu()
    if sigma is not None:
        sigma = sigma.cpu()
    mu = (mu*y_sd + y_mu)#return in the original units
    if sigma is None: 
        return mu 
    sigma = sigma*y_sd #return in the original units
    return mu, sigma

def unpack(result):
    return result if isinstance(result, tuple) else (result, None)