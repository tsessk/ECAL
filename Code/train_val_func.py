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

            metric_value = metric(np.nan_to_num(predictions.detach().numpy()), np.nan_to_num(y_batch.detach().numpy()))
            running_loss += loss.item() * X_batch.shape[0]
            running_metric += metric_value * X_batch.shape[0]
            if verbose:
                pbar.set_postfix({'loss': loss, 'MSE': metric_value})
        
        if scheduler is not None:
            scheduler.step()

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

            metric_value =  metric(np.nan_to_num(predictions.detach().numpy()), np.nan_to_num(y_batch.detach().numpy()))
            running_loss += loss.item() * X_batch.shape[0]
            running_metric += metric_value * X_batch.shape[0]
            if verbose:
                pbar.set_postfix({'loss': loss, 'MSE': metric_value})

        val_loss = running_loss / len(val_loader.dataset)
        val_metric = running_metric / len(val_loader.dataset)
        
        if val_metric < prev_met:
#             torch.save(model, f'model_acc_{val_metric:.3f}.pt')
            print('better metric =', val_metric)
            prev_met = val_metric

        wandb.log({'train_loss': train_loss,
                    'val_loss':  val_loss,
                    'train_metric': train_metric,
                   'val_metric': val_metric})
        
        if epoch % 10 == 0:
             torch.save(model, f'e_{epoch}_model.pt')
        
        

    print(f'Validation MSE: {val_metric:.3f}')
    
    return train_metric, val_metric