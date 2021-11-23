#!/usr/bin/env python3

#------------------------------------------------

def train(dataloader, model, loss_fn, optimizer):
    '''
    This single training loop makes the model predictions 
    on the training dataset (fed to it in batches), 
    and backpropagates the prediction error 
    to adjust the model’s parameters.
    '''
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute the prediction error:
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropogate the prediction error to adjust the model parameters:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f}  [{current:>5d}/{size:>5d}]')

#------------------------------------------------

def test(dataloader, model, loss_fn):
    '''
    check the model’s performance against the test dataset 
    to ensure it is learning.
    '''
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#------------------------------------------------

