import torch
from torch import nn
from sklearn.metrics import mean_squared_error
from torch.utils.data import DataLoader
import optuna



NUM_EPOCHS = 30
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
best_model = None
best_metric = 1e3

def build_model(trial, params):
    in_feats = len(tr_feats)
    num_layers = trial.suggest_int("num_layers", 1, 3)
    layers = []
    
    for i in range(num_layers):
        out_feats = trial.suggest_int('hidden_size_l{}'.format(i), 64, 256)
        
        layers.append(nn.Linear(in_features=in_feats, out_features=out_feats))
        layers.append(nn.BatchNorm1d(out_feats))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(p=params['dropout_p']))
        
        in_feats = out_feats
    
    layers.append(nn.Linear(in_features=in_feats, out_features=32))
    layers.append(nn.BatchNorm1d(32))
    layers.append(nn.ReLU())
    layers.append(nn.Linear(in_features=32, out_features=1))
    
    return nn.Sequential(*layers)
        
    
def train_and_evaluate(model, params, trial):
    criterion = nn.MSELoss()
    metric = lambda a, b: mean_squared_error(a, b, squared=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
    
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
        'dropout_p': trial.suggest_float('dropout_p', 0, 0.5)
    }
    
    model = build_model(trial, params)
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
study.optimize(objective, n_trials=250)
print('BEST PARAMETERS:')
print(study.best_params)