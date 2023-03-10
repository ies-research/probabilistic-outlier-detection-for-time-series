import os
import torch
import copy
import torch.nn as nn
from matplotlib import pyplot as plt
import numpy as np
from torch.nn import Parameter
from torch.autograd import Variable
from torchensemble import VotingRegressor
__all__ = ['neurons_dropout', 'WeightDropout', 'HeteroscedasticDropoutNetwork', 'Heteroscedastic_lstm', 'train']

## Standard Dropout
class MyStandardDropout(nn.Module):
    def __init__(self, p=0.2):
        super(MyStandardDropout, self).__init__()
        self.p = p
        # compensate_scaler is 1/(1-p). Set compensate_scaler to 0 when p=1 to avoid error.
        if self.p < 1:
            self.compensate_scaler = 1.0 / (1.0 - p)
        else:
            self.compensate_scaler = 0.0

    def forward(self, input):
        # if model.eval(), don't apply dropout
        if not self.training:
            return input

        # So that we have `input.shape` numbers of Bernoulli(1-p) samples
        mask_b = torch.rand(input.shape) > self.p
        mask_b = Variable(mask_b.type(torch.FloatTensor), requires_grad=False)

        # Multiply output by compensate_scaler
        return torch.mul(mask_b, input) * self.compensate_scaler



## Gaussian Dropout
class GaussianDropout(nn.Module):
    def __init__(self, alpha=1.0):
        print("<<<<<<<<< THIS IS DEFINETLY A GAUSSIAN DROPOUT TRAINING >>>>>>>>>>>>>>>")
        super(GaussianDropout, self).__init__()
        self.alpha = torch.Tensor([alpha])

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(1, alpha)
            epsilon = torch.randn(x.size()) * self.alpha + 1

            epsilon = Variable(epsilon)

            return x * epsilon
        else:
            return x


## Variational Dropout
class VariationalDropout(nn.Module):
    def __init__(self, alpha=1.0, dim=None):
        print("<<<<<<<<< THIS IS DEFINETLY A VARIATIONAL DROPOUT TRAINING >>>>>>>>>>>>>>>")
        super(VariationalDropout, self).__init__()

        self.dim = dim
        self.max_alpha = alpha
        # Initial alpha
        log_alpha = (torch.ones(dim) * alpha).log()
        self.log_alpha = nn.Parameter(log_alpha)

        self.nonlinearity = nn.Sigmoid()

    def kl(self):
        c1 = 1.16145124
        c2 = -1.50204118
        c3 = 0.58629921

        alpha = self.log_alpha.exp()

        negative_kl = 0.5 * self.log_alpha + c1 * alpha + c2 * alpha ** 2 + c3 * alpha ** 3

        kl = -negative_kl

        return kl.mean()

    def forward(self, x):
        """
        Sample noise   e ~ N(1, alpha)
        Multiply noise h = h_ * e
        """
        if self.train():
            # N(0,1)
            # epsilon = Variable(torch.randn(x.size()))
            epsilon = torch.randn(x.size())

            # Clip alpha
            self.log_alpha.data = torch.clamp(self.log_alpha.data, max=self.max_alpha)
            alpha = self.log_alpha.exp()

            # N(1, alpha)
            epsilon = epsilon * alpha + 1
            epsilon = Variable(epsilon)

            return x * epsilon
        else:
            return x


## Standout
def sample_mask(p):
    """Given a matrix of probabilities, this will sample a mask in PyTorch."""

    mask = Variable(torch.rand(p.size())) > p
    mask = mask.type(torch.FloatTensor)

    return mask

class StandoutDropout(nn.Module):

    def __init__(self, last_layer, alpha, beta):
        print("<<<<<<<<< THIS IS DEFINETLY A STANDOUT TRAINING >>>>>>>>>>>>>>>")
        super(StandoutDropout, self).__init__()
        self.pi = last_layer.weight
        self.alpha = alpha
        self.beta = beta
        self.nonlinearity = nn.Sigmoid()

    def forward(self, previous, current, p=0.5):

        self.p = self.nonlinearity(self.alpha * previous.matmul(self.pi.t()) + self.beta)
        self.mask = sample_mask(self.p)

        if self.train():
            return self.mask * current
        else:
            return (1 - self.p) * current


def _weight_drop(module, weights, dropout):
    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            w = torch.nn.functional.dropout(raw_w, p=dropout, training=module.training)
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)


class WeightDropout(torch.nn.Module):
    """
    The weight-dropped module applies recurrent regularization through a DropConnect mask on the
    hidden-to-hidden recurrent weights.
    Modified from PyTorch-NLP:
    <https://github.com/PetrochukM/PyTorch-NLP/bl ob/master/torchnlp/nn/weight_drop.py>.

    module (:class:`torch.nn.Module`): Containing module.
    weights (:class:`list` of :class:`str`): Names of the module weight parameters to apply a
    dropout too.
    dropout (float): The probability a weight will be dropped.

    """
    def __init__(self, module, weights, dropout=0.0):
        print("<<<<<<<<< THIS IS DEFINETLY A WEIGHT DROPOUT TRAINING >>>>>>>>>>>>>>>")
        super(WeightDropout, self).__init__()
        _weight_drop(module, weights, dropout)
        self.forward = module.forward


def neurons_dropout(p=None, alpha=None, beta=None, dim=None, last_layer=None, method='standard'):
    if method == 'dropout':
        return nn.Dropout(p)
    elif method == 'standard':
        return MyStandardDropout(p)
    elif method == 'gaussian':
        return GaussianDropout(p/(1-p))
    elif method == 'variational':
        return VariationalDropout(p/(1-p), dim)
    elif method == 'standout':
        return StandoutDropout(last_layer, alpha, beta)


class HeteroscedasticDropoutNetwork(nn.Module):
    def __init__(self,
                 input_size,
                 ann_structure,
                 output_size,
                 activation_f=nn.LeakyReLU(),
                 dropout_method='dropout',
                 dropout_rate=0.1,
                 alpha=None,
                 beta=None,
                 weight_drop=0):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation_f = activation_f
        self.dropout_method = dropout_method
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.beta = beta
        self.weight_drop = weight_drop
        self.ann_structure = copy.deepcopy(ann_structure)
        self.ann_structure.insert(0, input_size)

        self.model = []
        for in_, out_ in zip(self.ann_structure[0:-1], self.ann_structure[1:]):
            # use weight-dropout method
            if self.weight_drop:
                model = WeightDropout(nn.Linear(in_, out_), ['weight'], dropout=self.weight_drop)
                self.model.append(model)
                if self.activation_f is not None:
                    self.model.append(self.activation_f)
                    # use node-dropout methods
            else:
                self.model.append(nn.Linear(in_, out_))
                if self.activation_f is not None:
                    self.model.append(self.activation_f)
                # add different neurous-dropout methods
                self.model.append(neurons_dropout(self.dropout_rate,
                                                  self.alpha,
                                                  self.beta,
                                                  out_,
                                                  nn.Linear(in_, out_),
                                                  self.dropout_method))

        self.model.append(nn.Linear(self.ann_structure[-1], self.output_size))
        self.model = nn.Sequential(*self.model)
        # self.ann_structure = ann_structure

    def forward(self, x):
        if self.dropout_method == 'standout':
            # first fc
            previous = x
            current = self.model[1](self.model[0](x))
            x = self.model[2](previous, current)
            # second fc
            previous = x
            current = self.model[4](self.model[3](x))
            x = self.model[5](previous, current)
            # final layer
            x = self.model[6](x)
        else:
            x = self.model(x)

        return x

    def kl(self):
        kl = 0
        for name, module in self.model.named_modules():
            if isinstance(module, VariationalDropout):
                kl += module.kl().sum()
        return kl

    def predict_prob(self, x, iters, p=None):
        """
        Using MC dropoout to estimate the uncertainty of input x
        x: array-like input, [n_samples, n_features] or [n_samples*n_timestep, n_features]
        iters: the random forward iteration to get the uncertainty
        p: float, the dropout rate, default as None if the rate is not changed.
        :return:
        Yt_hat: T stochastic forward passes through the network
        MC_pred: MC-Dropout prediction, i.e., averaging results of Yt_hat
        pred_var: Predictive variance
        """
        mode = self.training

        with torch.no_grad():
            Yt_hat = torch.stack([self.forward(x) for _ in range(iters)], dim=-1)

        sampled_mus = Yt_hat[:, 0, :]
        mean_mus = sampled_mus.mean(dim=-1)
        noises = torch.exp(Yt_hat[:, 1, :])

        # calculating different uncertainties
        aleatoric = (noises ** 2).mean(axis=-1) ** 0.5
        epistemic = (sampled_mus.std(axis=-1))

        self.train(mode=mode)

        return Yt_hat, sampled_mus, mean_mus, noises, aleatoric, epistemic

    # The following two functions need to be adapted
    def enable_dropout(self):
        """ Function to enable the dropout layers during test-time """
        for m in self.model.modules():
            if m.__class__.__name__[-7:].startswith('Dropout'):
                m.train()

    def set_dropout(self, p):
        '''
        set dropout rate of each layer.
        :param p: float, dropout rate
        :return:
        '''
        pass

class VotingRegressor_modified(VotingRegressor):

    def forward(self, x):
        # Average of predictions from all base estimators.
        outputs = [estimator(x) for estimator in self.estimators_]
        return outputs

    def predict_prob(self, x, iters):
        mode = self.training

        with torch.no_grad():
            Yt_hat = torch.stack(self.forward(x), dim=-1)

        sampled_mus = Yt_hat[:, 0, :]
        mean_mus = sampled_mus.mean(dim=-1)
        noises = torch.exp(Yt_hat[:, 1, :])

        # calculating different uncertainties
        aleatoric = (noises ** 2).mean(axis=-1) ** 0.5
        epistemic = (sampled_mus.std(axis=-1))

        self.train(mode=mode)

        return Yt_hat, sampled_mus, mean_mus, noises, aleatoric, epistemic

class Heteroscedastic_lstm(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_first=True, dropout=0.1, proj_size=0):
        '''
        if proj_size is 0, a linear layer is added as an output layer. Otherwise, only lstm model is generated.
        '''
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size * 2
        self.proj_size = proj_size if proj_size == 0 else self.output_size
        self.batch_first = batch_first
        self.dropout = dropout

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                            batch_first=batch_first, dropout=dropout, proj_size=self.proj_size)
        if self.proj_size != self.output_size:
            self.output_layer = nn.Linear(in_features=self.hidden_size, out_features=self.output_size)

    def forward(self, X, h_0=None, c_0=None):
        batch_size = X.shape[0] if self.batch_first else X.shape[1]

        if h_0 is None and c_0 is None:
            c_0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
            if self.proj_size == 0:
                h_0 = torch.randn(self.num_layers, batch_size, self.hidden_size)
                y_pred, (_, _) = self.lstm(X, (h_0, c_0))
                y_pred = nn.output_layer(y_pred)
            else:
                h_0 = torch.randn(self.num_layers, batch_size, self.proj_size)
                # print(f'X_shape: {X.shape}')
                y_pred, (_, _) = self.lstm(X, (h_0, c_0))

        if self.batch_first:
            return y_pred[:, -1, :]
        else:
            return y_pred[-1, :, :]


def ensemble_uncertainty_estimation(ensemble_model, n_estimator, x_tensor):
    preds = []
    mode = ensemble_model.training
    ensemble_model.eval()
    with torch.no_grad():
        for estimator in ensemble_model.estimators_:
            preds.append(estimator.forward(x_tensor))

    preds = torch.stack(preds, 0)
    output_size = int(preds.shape[-1] / 2)

    epistemic_uncertainty = preds[:, :, :output_size].std(dim=0)
    noises = torch.exp(preds[:, :, output_size:])
    aleatoric_uncertainty = (noises ** 2).mean(dim=0).sqrt()

    ensemble_model.train(mode=mode)

    return epistemic_uncertainty, aleatoric_uncertainty


def train(model, criterion, optimizer, dataloader, epochs, X_train_torch, y_train_torch, X_val_torch, y_val_torch, plot_path,
          phase, serial_number):
    # Loss storage
    losses = []
    val_losses = []

    train_mses = []
    val_mses = []

    # Training process
    for epoch_idx in range(epochs):
        running_loss = []
        model.train()
        for batch_idx, sample in enumerate(dataloader):
            optimizer.zero_grad()
            y_pred = model.forward(sample[0])
            # print(y_pred)
            # print(y_pred.shape)
            loss = criterion(y_pred, sample[1])

            loss.backward()
            optimizer.step()

            running_loss.append(loss.item())

        with torch.no_grad():
            train_mses.append(nn.MSELoss()(model.forward(X_train_torch)[:, :1], y_train_torch).item())
        losses.append(np.mean(running_loss))

        model.eval()
        y_pred_val = model.forward(X_val_torch)
        val_loss = criterion(y_pred_val, y_val_torch).item()
        with torch.no_grad():
            val_mses.append(nn.MSELoss()(model.forward(X_val_torch)[:, :1], y_val_torch).item())

        val_losses.append(val_loss)

        if epoch_idx % 10 == 0 or epoch_idx == epochs - 1:
            print(
                f'{epoch_idx + 1}/{epochs}, training_loss={losses[-1]:.3f}, valid_loss={val_loss:.3f}, MSE train: {train_mses[-1]:.3f}, MSE valid: {val_mses[-1]:.3f}')

    if plot_path:
        figs, axes = plt.subplots(1, 2, figsize=(16, 4))
        axes[0].plot(losses, label='training')
        axes[0].legend()
        axes[0].set_title(f'Learning curve of inverter_{str(serial_number)})')
        axes[1].plot(val_losses, label='validation')
        axes[1].legend()
        axes[1].set_title(f'Learning curve of inverter_{str(serial_number)})')
        save_dir = os.path.join(plot_path,
                                'inverter_' + str(serial_number))
        store_path_nll = save_dir + '/' + 'nll'+ '_' + 'inverter_' + str(str(serial_number)) + '.png'
        plt.savefig(store_path_nll, dpi=300)
        plt.show()

        plt.figure(figsize=(16, 4))
        plt.plot(train_mses, label='Training')
        plt.plot(val_mses, label='Validation')
        plt.legend()
        plt.title(f'{phase} phase of inverter_{str(serial_number)}')
        store_path = save_dir + '/' + 'mse' + '_' + 'inverter_' + str(str(serial_number)) + '.png'
        plt.savefig(store_path, dpi=300)
        plt.show()

    return model