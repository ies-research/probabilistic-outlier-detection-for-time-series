from scipy.stats import chi2
from scipy.stats import norm
from sklearn.ensemble import IsolationForest
import warnings
import numpy as np

warnings.filterwarnings("ignore")


class mahalanobis_model:
    def __init__(self, ppf=0.999):
        self.ppf = ppf

    def fit(self, train_data):
        """
        train_data: numpy array with shape (N,M)   N: data length, M: channel num
        """
        if train_data.shape[1] == 1:
            mean = np.abs(train_data).reshape(-1).mean()
            std = np.abs(train_data).reshape(-1).std()
            self.threshold = norm.ppf(self.ppf, mean, std)
        else:
            self.error_cov = np.cov(train_data.T)
            self.threshold = np.sqrt(chi2.ppf([self.ppf], train_data.shape[1]))

    def outlier_score(self, test_data):
        """
        test_data: numpy array with shape (N,M)   N: data length, M: channel num
        return:
        outlier_score: numpy array with shape (N)
        """
        if test_data.shape[1] == 1:
            mahalanobis = np.abs(test_data).reshape(-1)
        else:
            mahalanobis = np.sqrt(np.sum((test_data @ np.linalg.inv(self.error_cov)) * test_data, axis=1))
        return mahalanobis

    def outlier_result(self, test_data):
        """
        test_data: numpy array with shape (N,M)   N: data length, M: channel num
        return:
        outlier_result: numpy array with shape (N), return true(outlier) or false(normal)
        """
        return self.outlier_score(np.abs(test_data)) > self.threshold


class isolationforest_model:
    def __init__(self, n_estimators=100, random_state=42, ppf=0.999):
        self.iforest = IsolationForest(random_state=random_state, n_estimators=n_estimators)
        self.ppf = ppf

    def fit(self, train_data):
        """
        train_data: numpy array with shape (N,M)   N: data length, M: channel num
        """
        self.iforest.fit(train_data)
        scores = self.outlier_score(train_data)
        self.threshold = norm.ppf(self.ppf, scores.mean(),
                                  scores.std())  # np.sqrt(chi2.ppf([0.999], train_data.shape[1]))

    def outlier_score(self, test_data):
        """
        test_data: numpy array with shape (N,M)   N: data length, M: channel num
        return:
        outlier_score: numpy array with shape (N)
        """
        scores = self.iforest.score_samples(test_data)
        return np.abs(scores)

    def outlier_result(self, test_data):
        """
        test_data: numpy array with shape (N,M)   N: data length, M: channel num
        return:
        outlier_result: numpy array with shape (N), return true(outlier) or false(normal)
        """
        return self.outlier_score(test_data) > self.threshold
