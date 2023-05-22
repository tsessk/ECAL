# !pip install pytorch-tabnet
from pytorch_tabnet.tab_model import TabNetRegressor

X_test_n = X_test.copy()
p_ECAL = X_test_n['p_ECAL']
cell_size = X_test_n['cell_size']
t_ECAL = X_test_n['t_ECAL']
X_test_n.drop(columns=['t_ECAL', 'p_ECAL', 'cell_size', 't_pred'], inplace=True)

clf = TabNetRegressor(scheduler_fn=torch.optim.lr_scheduler.CyclicLR,
                     scheduler_params={'base_lr': 2e-2, 'max_lr': 0.01})

clf.fit(X_train.values, y_train_n,
       eval_set=[(X_val.values, y_val_n)],
       eval_metric=['rmse'])

preds = clf.predict(X_test_n.values)
preds = preds.reshape(-1, )

X_test_n['p_ECAL'] = p_ECAL
X_test_n['cell_size'] = cell_size
X_test_n['t_ECAL'] = t_ECAL

# baseline mean_RMSE t_pred_xgb: 0.02712, std_RMSE: 0.00261
mean_RMSE, std_RMSE = get_rmse_metric(y_test.values, preds)
print('mean_RMSE t_pred_NN: {:.5f}, std_RMSE: {:.5f}'.format(mean_RMSE, std_RMSE))

X_test_n['t_pred'] = preds
rec_title = f'TabNet_def_ReduceLROnPlateau'
sigmas, sigmas_std = extract_timing_resolution(X_test_n, save=True, title=rec_title)
plot_rec_curve(sigmas, sigmas_std, save=True, title=rec_title)

