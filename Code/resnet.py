import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
import optuna


class ResNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.ModuleList(
            [
                self.build_block(len(tr_feats), 128),
                self.build_block(128, 256),
                self.build_block(256, 128),
                self.build_block(128, 64)
            ]
        )
        
        self.skip_connections = nn.ModuleList(
            [
                nn.AdaptiveAvgPool1d(128),
                nn.AdaptiveAvgPool1d(256),
                nn.AdaptiveAvgPool1d(128),
                nn.AdaptiveAvgPool1d(64)
            ]
        )
        self.head = nn.Linear(64, 1)
        
    def build_block(self, in_feats, out_feats):
        return  nn.Sequential(
                    nn.Linear(in_feats, out_feats),
                    nn.BatchNorm1d(out_feats),
                    nn.ReLU())
        
        
    def forward(self, x):
        out = x

        for i in range(len(self.layers)):
            out = self.skip_connections[i](out) + self.layers[i](out)
            
        out = self.head(out)
        return out


NUM_EPOCHS = 25
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
best_model = None
best_metric = 1e3

    
def train_and_evaluate(model, params, trial):
    
    TrainLoader = DataLoader(TrainSet, batch_size=params['batch_size'], pin_memory=True, num_workers=2)
    ValLoader = DataLoader(ValSet, batch_size=params['batch_size'], pin_memory=True, num_workers=2)
    
    criterion = nn.MSELoss()
    metric = lambda a, b: mean_squared_error(a, b, squared=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS * len(TrainLoader))

    
    if torch.cuda.is_available():
        criterion = criterion.cuda()
        model = model.cuda()
    
    print(next(model.parameters()).device)
    
    for epoch in tqdm(range(NUM_EPOCHS)):
        model.train()
        running_loss, running_metric = 0, 0
        
        for X_batch, y_batch in TrainLoader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            
            predictions = model(X_batch)
            y_batch = y_batch.unsqueeze(1)
            loss = criterion(predictions, y_batch)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            metric_value = metric(np.nan_to_num(predictions.cpu().detach().numpy()), np.nan_to_num(y_batch.cpu().detach().numpy()))
            running_loss += loss.item() * X_batch.shape[0]
            running_metric += metric_value * X_batch.shape[0]
            train_loss = running_loss / len(TrainLoader.dataset)
            train_metric = running_metric / len(TrainLoader.dataset)
            
            
        with torch.no_grad():
            model.eval()
            running_loss, running_metric = 0, 0
            for X_batch, y_batch in ValLoader:

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                predictions = model(X_batch)
                y_batch = y_batch.unsqueeze(1)
                loss = criterion(predictions, y_batch)
                metric_value =  metric(np.nan_to_num(predictions.cpu().detach().numpy()), np.nan_to_num(y_batch.cpu().detach().numpy()))
                running_loss += loss.item() * X_batch.shape[0]
                running_metric += metric_value * X_batch.shape[0]
                
        val_loss = running_loss / len(ValLoader.dataset)
        val_metric = running_metric / len(ValLoader.dataset)
        
        trial.report(val_metric, epoch)
        
        if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
                
    return val_metric

    

def objective(trial):
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-1),
        'batch_size': trial.suggest_categorical('batch_size', [1024, 2048, 4096])
    }
    
    model = ResNet()
    rmse = train_and_evaluate(model, params, trial)
    print(trial.params)
    
    global best_model
    global best_metric
        
    if rmse < best_metric:
        best_metric = rmse
        best_model = model
        print('New best')

    return rmse

study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(), pruner=optuna.pruners.MedianPruner())
study.optimize(objective, n_trials=20)

print('BEST PARAMETERS:')
print(study.best_params)

        
