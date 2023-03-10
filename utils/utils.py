import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def gaussian_nll_loss(output, target):
    # here using torch.exp() to calcualte variance instead of the original output
    # because the lower boundary of given scale in torch.distributions.Normal is 0.0
    mu, sigma = output[:, :1], torch.exp(output[:, 1:])
    dist = torch.distributions.Normal(mu, sigma)
    loss = -dist.log_prob(target)

    return loss.sum()

def dataset(features, targets, val_size, batch_size, method):
    # Training/validation dataset
    X_train, X_val, y_train, y_val = train_test_split(features, targets, test_size=val_size, shuffle=True)
    X_train_torch, y_train_torch = torch.Tensor(X_train.values), torch.Tensor(y_train.values)
    X_val_torch, y_val_torch = torch.Tensor(X_val.values), torch.Tensor(y_val.values)
    if method == 'LSTM':
        X_train_torch = X_train_torch.reshape(-1, 24, 5)
        X_val_torch = X_val_torch.reshape(-1, 24, 5)
    train_dataset = TensorDataset(X_train_torch, y_train_torch)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return X_train_torch, y_train_torch, X_val_torch, y_val_torch, dataloader

