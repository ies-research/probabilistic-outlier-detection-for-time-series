import pandas as pd
import torch
import warnings
import numpy as np
from scipy import stats
import properscoring as ps

warnings.filterwarnings("ignore")


class AlarmSystem:
    def __init__(self, pred_model, pred_window_size=pd.Timedelta(days=1), history_window_size=pd.Timedelta(days=10)):
        """
        pred_model: object, probabilistic forecasting models
        pred_window_size: pd.Timedelta, windows size of prediction, for example next 1 days data
        history_window_size: pd.Timedelta, windows size for analysis historical data, e.g.calibration_score,
        """
        self.pred_model = pred_model
        self.pred_window_size = pred_window_size
        self.history_window_size = history_window_size

    def pred_win_target(self, pred_win_input, sample_num=10):
        """
        pred_win_input: numpy array with shape (N,L),  N: data length, L: input channel num
        sample_num: int, num of sampel to get post distribution
        return:
        Yt_hat_pred: numpy array with shape (N,M, Q)   N: data length, M: output channel num, Q: num of samples
        y_pred: numpy array with shape (N,M),    N: data length, M: channel num, sampled predictions by dropout
        aleatoric_pred: numpy array with shape (N,M),    N: data length, M: channel num, aleatoric_uncertainty for prediction
        epistemic_pred: numpy array with shape (N,M),    N: data length, M: channel num, epistemic_uncertainty for prediction
        """

        predictions = self.pred_model.predict_prob(torch.tensor(pred_win_input).float(), iters=sample_num)
        Yt_hat_pred = predictions[0]
        sampled_mus_pred = predictions[1]
        y_pred = predictions[2]
        noises_pred = predictions[3]
        aleatoric_pred = predictions[4]
        epistemic_pred = predictions[5]
        return Yt_hat_pred, y_pred, aleatoric_pred, epistemic_pred

    def calibration_score_(self, s1, s2):
        '''
        s1: np.array, actual quantiles
        s2: np.array, prediction quantiles

        return:
        sizes: list, each element describes the size of each section between two quatiles.
        '''
        sizes = []

        for i in range(10):
            h = s1[i + 1] - s1[i]
            if (s1[i + 1] - s2[i + 1]) * (s1[i] - s2[i]) < 0:
                h1 = h / ((np.abs(s1[i + 1] - s2[i + 1]) / np.abs(s1[i] - s2[i])) + 1)
                h2 = h - h1
                size = np.abs(s1[i] - s2[i]) * h1 * 0.5 + np.abs(s1[i + 1] - s2[i + 1]) * h2 * 0.5

            else:
                size = 0.5 * h * np.abs(s1[i] + s1[i + 1] - s2[i] - s2[i + 1])

            sizes.append(size)
        return np.array(sizes)

    def calibration_score(self, real_win_target, qpreds_win_target, quantiles):
        """
        real_win_target: numpy array with shape (N,M),  N: data length, M: output channel num, the real value for prediction window
        qpreds_win_target: numpy array with shape (N,Q,M),  N: data length, Q: num of quantiles M: output channel num, a window of data for prediction
        quantiles: numpy.array with shape(Q), Q: num of quantiles
        return:
        quantile_scores: list with Q element  Q: num of quantiles
        calibration_score: int,  calibration_score for this window of data
        """
        assert qpreds_win_target.shape[0] == len(real_win_target)
        assert qpreds_win_target.shape[1] == len(quantiles)

        quantile_scores = [
            ((qpreds_win_target[:, q] - real_win_target).sum(axis=1) >= 0).sum() / len(qpreds_win_target)
            # (qpreds_win_target[:, q, 0] >= real_win_target[:,0]).sum() / len(qpreds_win_target)
            for q in range(len(quantiles))
        ]
        # print(qpreds_win_target.shape, real_win_target.shape)
        calibration_score = self.calibration_score_(quantile_scores, quantiles).sum()
        return quantile_scores, calibration_score

    def crp_score(self, history_win_target, qpreds_win_target):
        """
        history_win_target: numpy array with shape (N,M),  N: data length, M: output channel num, the real value for a history window
        qpreds_win_target: numpy array with shape (N,Q,M),  N: data length, Q: num of quantiles M: output channel num,  prediction value for a history window
        quantiles: numpy.array with shape(Q), Q: num of quantiles
        return:
        crps: numpy array with shape (N,M)   N: data length, M: output channel num
        """
        # print(history_win_target.shape, qpreds_win_target.shape)
        crps = ps.crps_ensemble(history_win_target, np.transpose(qpreds_win_target, (0, 2, 1)))
        # observations.shape not in [forecasts.shape, forecasts.shape[:-1]]
        # print(history_win_target.shape, qpreds_win_target.shape, qpreds_win_target.shape[:-1])
        return crps

    def create_quantiles(self, n, min_q=0.01, max_q=0.99):
        """
        n: int, num of quantiles
        min_q: float, min_q
        max_q: float, max_q
        return:
        quantiles: numpy.array with shape(N), N: num of quantiles
        """
        n -= 2  # because we add the lowest and highest quantiles manually
        n_equally_spaced = np.linspace(1 / (n + 1), 1 - 1 / (n + 1), n)
        quantiles = np.concatenate([np.array([min_q]),
                                    n_equally_spaced,
                                    np.array([max_q])])
        return quantiles

    def postprocess_samples(self, samples):
        """
        samples: numpy.array with shape(M, N, Q), M: data length, N:num of output channel * 2, Q: num of samples
        return:
        mus: numpy.array with shape(M, N/2), M: data length N/2: num of channel
        sigmas: numpy.array with shape(M, N/2), M: data length N/2: num of channel
        """
        mus = torch.mean(samples[:, :int(samples.shape[1] / 2), :], axis=-1)
        sigmas = torch.mean(torch.exp(samples[:, int(samples.shape[1] / 2):, :]), axis=-1)
        return mus.numpy(), sigmas.numpy()

    def quantiles_preds(self, pred_mu, pred_sigma, quantiles, num_sample=10):
        """
        pred_mu: numpy array with shape (N,M),  N: data length, M: output channel num
        pred_sigma: numpy array with shape (N,M),  N: data length, M: output channel num,
        quantiles: numpy.array with shape(Q), Q: num of quantiles
        return:
        qpreds: numpy array with shape (N,Q,M)   N: data length, Q: num of quantiles,  M: output channel num
        """
        qpreds = []
        for i in range(len(pred_mu)):
            if pred_mu.shape[1] == 1:
                pred_norm = stats.norm.rvs(pred_mu[i, 0], pred_sigma[i, 0], size=num_sample)
            else:
                pred_norm = stats.multivariate_normal.rvs(pred_mu[i, :], np.diag(pred_sigma[i, :]), size=num_sample)
            qs_from_posterior = np.quantile(pred_norm, quantiles, axis=0)
            qpreds.append(qs_from_posterior)
        if pred_mu.shape[1] == 1:
            return np.expand_dims(np.array(qpreds), axis=2)
        return np.array(qpreds)
