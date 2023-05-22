import wandb
import torch
from torch import nn

NUM_EPOCHS = 12
models = list()

for i in range(20):
    model = nn.Sequential(
    nn.Linear(in_features=len(tr_feats), out_features=1)
    )
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    model = model.to(device)
    scheduler = None #torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    metric_rmse = lambda a, b: mean_squared_error(a, b, squared=False)

    run_config = {
        "learning_rate": 1e-3,
        "epochs": NUM_EPOCHS,
        "optimizer": optimizer,
        "scheduler": 'None'
    }

    run = wandb.init(project="HSE_ECAL_project", config=run_config)
    
    train_and_validate(model, optimizer, criterion, metric_rmse, TrainLoader, ValLoader, NUM_EPOCHS, device, scheduler, verbose=True)
    torch.save(model, f'ensemble_1l_{i}_model.pt')
    models.append(model)
    run.finish()


ensemble_preds = np.zeros(y_test.shape)

for m in models:
    preds = []
    for data, target in TestLoader:
        data = data.to(device)
        preds.append(m(data).item())
    preds = np.array(preds)
    ensemble_preds += preds

ensemble_preds /= len(models)

mean_RMSE, std_RMSE = get_rmse_metric(y_test.values, ensemble_preds)
print('mean_RMSE t_pred_NN: {:.5f}, std_RMSE: {:.5f}'.format(mean_RMSE, std_RMSE))


rec_title = f'ENSEMBLE_1L_20'
X_test['t_pred'] = ensemble_preds
sigmas, sigmas_std = extract_timing_resolution(X_test, save=True, title=rec_title)
plot_rec_curve(sigmas, sigmas_std, save=True, title=rec_title)
    