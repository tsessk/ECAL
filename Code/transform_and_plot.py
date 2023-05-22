import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wandb
from lmfit.models import GaussianModel
import matplotlib.pyplot as plt


def get_time_cell_3x3(row):
    time = row[:9]
    energy = row[9:]
    return (time * energy).sum() / (energy.sum() + 1e-6)


def get_time_cell_5x5(row):
    time = row[:25]
    energy = row[25:]
    return (time * energy).sum() / (energy.sum() + 1e-6)

def get_new_feats_and_train_feats(df, layers=[0], conv_check=False):
    df['t_ECAL'] = df.timing.values

    for layer in layers:
        df['t{}_weighted_3x3'.format(layer)] = np.apply_along_axis(get_time_cell_3x3, 
                                                    axis=1, 
                                                    arr=df[['t{}_{}'.format(layer,c) for c in [6,7,8,11,12,13,16,17,18]] + \
                                                           ['l{}_{}'.format(layer,c) for c in [6,7,8,11,12,13,16,17,18]]].values)

        df['t{}_weighted_5x5'.format(layer)] = np.apply_along_axis(get_time_cell_5x5, 
                                                    axis=1, 
                                                    arr=df[['t{}_{}'.format(layer,c) for c in range(25)] + \
                                                           ['l{}_{}'.format(layer,c) for c in range(25)]].values)


    for layer in layers:
        df['l{}_sum_3x3'.format(layer)] = \
            df[['l{}_{}'.format(layer, i) for i in [6,7,8,
                                                    11,12,13,
                                                    16,17,18]]].values.sum(axis=1)

        df['l{}_sum_5x5'.format(layer)] = \
            df[['l{}_{}'.format(layer, i) for i in range(25)]].values.sum(axis=1)
        
    train_features = []
    
    if not conv_check:
        train_features += ['t0_{}'.format(i) for i in range(25)] + \
                          ['t0_weighted_3x3','t0_weighted_5x5']
    
    for layer in layers:
            train_features += ['l{}_{}'.format(layer, i) for i in range(25)]
            if not conv_check:
                train_features += ['l{}_sum_3x3'.format(layer)] + \
                ['l{}_sum_5x5'.format(layer)]
    
    return df, train_features


class MyDataset(Dataset):
    def __init__(self, X_, y_):
        self.X = torch.tensor(X_.values,dtype=torch.float32)
        self.y = torch.tensor(y_.values,dtype=torch.float32)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index],self.y[index]

class MyConvDataset1Layer(Dataset):
    def __init__(self, X_, y_):
        super(Dataset, self).__init__()
        self.X = torch.tensor(X_.values,dtype=torch.float32)
        self.y = torch.tensor(y_.values,dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index].reshape((5, 5)).unsqueeze(0), self.y[index]

class MyConvDataset2Layer(Dataset):
    def __init__(self, X_, y_):
        super(Dataset, self).__init__()
        self.X = torch.tensor(X_.values,dtype=torch.float32)
        self.y = torch.tensor(y_.values,dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index].reshape((5, 10)).unsqueeze(0), self.y[index]


def train_and_validate(model, optimizer, criterion, metric, train_loader, val_loader,
                       num_epochs, device, scheduler=None, verbose=True):
    
    prev_met = 10e3


    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, running_metric = 0, 0
        
        pbar = tqdm(train_loader, desc=f'Training {epoch}/{num_epochs}') \
            if verbose else train_loader

        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            predictions = model(X_batch)
            y_batch = y_batch.unsqueeze(1)
            loss = criterion(predictions, y_batch)
            loss.backward()
            optimizer.step()
            
            if scheduler is not None:
                scheduler.step()

            metric_value = metric(np.nan_to_num(predictions.cpu().detach().numpy()), np.nan_to_num(y_batch.cpu().detach().numpy()))
            running_loss += loss.item() * X_batch.shape[0]
            running_metric += metric_value * X_batch.shape[0]
            if verbose:
                pbar.set_postfix({'loss': loss, 'MSE': metric_value})
        

        train_loss = running_loss / len(train_loader.dataset)
        train_metric = running_metric / len(train_loader.dataset)

        model.eval()
        running_loss, running_metric = 0, 0
        pbar = tqdm(val_loader, desc=f'Validating {epoch}/{num_epochs}') \
            if verbose else val_loader

        for X_batch, y_batch in pbar:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            predictions = model(X_batch)
            y_batch = y_batch.unsqueeze(1)
            loss = criterion(predictions, y_batch)
            
            metric_value =  metric(np.nan_to_num(predictions.cpu().detach().numpy()), np.nan_to_num(y_batch.cpu().detach().numpy()))
            running_loss += loss.item() * X_batch.shape[0]
            running_metric += metric_value * X_batch.shape[0]
            if verbose:
                pbar.set_postfix({'loss': loss, 'MSE': metric_value})

        val_loss = running_loss / len(val_loader.dataset)
        val_metric = running_metric / len(val_loader.dataset)
        
        if val_metric < prev_met:
            print('better metric =', val_metric)
            prev_met = val_metric

        wandb.log({'train_loss': train_loss,
                    'val_loss':  val_loss,
                    'train_metric': train_metric,
                   'val_metric': val_metric})
        
        

    print(f'Validation MSE: {val_metric:.3f}')
    
    return train_metric, val_metric

def get_rmse(x, y):
    return np.nanmean(((x - y) ** 2.0)) ** 0.5

def get_rmse_metric(x, y, folds=5):
    if x.shape != y.shape:
        print('x.shape != y.shape')
        raise

    splits = np.array_split(np.arange(len(x)), folds)

    rmse = []
    for split in splits:
        rmse.append(get_rmse(x[split], y[split]))

    return np.nanmean(rmse), np.nanstd(rmse)


def extract_timing_resolution(df, save=True, title=None):
    cell_size = 1.515
    mod = GaussianModel()
    E_bins = [10000., 30000., 50000., 70000., 90000.]

    sel = (df.cell_size == cell_size)
    df = df[sel & (df.p_ECAL > E_bins[0]) & (df.p_ECAL < E_bins[-1])].reset_index(drop=True)
    target = df.t_pred
    n_stds = 5

    range_all = (-n_stds*np.std(target - df.t_ECAL), n_stds*np.std(target - df.t_ECAL))

    fig, ax = plt.subplots(1, len(E_bins)-1, figsize=(20,3))
    bins_plotting = (np.array(E_bins[:-1]) + np.array(E_bins[1:])) / 2.0
    sigmas = []
    sigmas_std = []

    for i in [0, 1, 2, 3]:
        mask = (df.p_ECAL >= E_bins[i]) & (df.p_ECAL < E_bins[i+1])
        bin_heights, bin_borders, _ = ax[i].hist(target[mask]/df.t_ECAL[mask] - 1, bins=30, \
             range=(-n_stds*np.std(target[mask]/df.t_ECAL[mask] - 1), n_stds*np.std(target[mask]/df.t_ECAL[mask] - 1)), color='C02')
        bin_centers = (bin_borders[:-1] + bin_borders[1:]) / 2.0
        pars = mod.guess(bin_heights, x = bin_centers)
        out = mod.fit(bin_heights, pars, x = bin_centers)
        x = np.linspace(bin_borders[0], bin_borders[-1], 200)
        ax[i].plot(bin_centers, out.best_fit, label='best fit', color='C03')
        ax[i].set_title(f'Bin {i} ({0.001*E_bins[i]}-{0.001*E_bins[i+1]} GeV, {len(df.p_ECAL[mask])} entries)')
        ax[i].set_xlabel(r'$t_{rec} - t_{gen}$ [ns]', fontsize=14)

        
        sigmas.append(out.params['sigma'].value)
        sigmas_std.append(out.params['sigma'].stderr)
    
    if title is not None:
        plt.title(title)
    if save:
        plt.savefig(f'plot_hist_{title}.png', dpi=300, bbox_inches='tight')
    plt.close()
        
    return sigmas, sigmas_std

def plot_rec_curve(sigmas, sigmas_std, save=True, title=None):
    fig, ax = plt.subplots(figsize=(10,10))

    x = [20, 40, 60, 80]

    x_ref = [2., 3., 4., 5., 20., 30., 50., 70., 100.]
    y_ref1 = [44.71,34.90,29.48,26.19,19.39,17.40,15.35,14.04,13.63]
    y_ref2 = [56.99,47.04,41.35,37.89,26.15,22.83,16.34,13.36,12.36]
    
    if len(sigmas) == 0 or len(sigmas_std) == 0 or None in sigmas or None in sigmas_std:
        return
    
    y_b = [i*100000 for i in sigmas]
    yerr_b = [i*100000 for i in sigmas_std]

    ax.set_xlabel(r'$E_{gen}$ [GeV]', fontsize=18)
    ax.set_ylabel('Time resolution [ps]', fontsize=18)

    
    ax.grid(True, which="both")

    ax.errorbar(x, y_b, yerr=yerr_b, fmt='o', c='C03', alpha=0.7, label='Current reconstruction')
    ax.plot(x_ref, y_ref1, 'g--', label="Reference 1")
    ax.plot(x_ref, y_ref2, 'b--', label="Reference 2")
    ax.legend(fontsize=16, loc='upper right')
    
    if title is not None:
        plt.title(title)
    
    if save:
        plt.savefig(f'plot_rec_{title}.png', dpi=300, bbox_inches='tight')
    plt.close()