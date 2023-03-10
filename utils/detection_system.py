import pandas as pd
import warnings
import numpy as np
from sklearn.cluster import DBSCAN
from tslearn.metrics import dtw
from scipy.stats import norm
from scipy import stats

warnings.filterwarnings("ignore")
from .preprocessing import DataPreprocess


class DetectionSystem:
    def __init__(self, pred_model, detect_model, pred_window_size=pd.Timedelta(days=1), real_failure_list=None):
        """
        pred_model: object, probabilistic forecasting models
        detect_model: object, outlier detection models
        pred_window_size: pd.Timedelta, windows size of prediction, for example next 1 days data
        real_failure_list: pd.Dataframe, real_failure_list
        """
        self.pred_model = pred_model
        self.detect_model = detect_model
        self.pred_window_size = pred_window_size
        self.real_failure_list = real_failure_list
        self.data_process = DataPreprocess()
        self.real_failure_list = real_failure_list

    def outlier_score(self, window):
        """
        window: numpy.array with shape(N, M) N: len of data, M: num of output channel
        return:
        outlier_score: numpy.array with shape(N,1), N: len of data
        """
        return self.detect_model.outlier_score(window)

    def outlier_prob(self, real_output_window, pred_window_mean, pred_window_std):
        """
        real_output_window: numpy.array with shape(N, M) N: len of data, M: num of output channel
        pred_window_mean: numpy.array with shape(N, M) N: len of data, M: num of output channel
        pred_window_std: numpy.array with shape(N, M) N: len of data, M: num of output channel
        return:
       outlier_prob: numpy.array with shape(N,1), N: len of data
        """
        if real_output_window.shape[1] == 1:
            sampled_n = 10
            post_pred_tmp = np.zeros((real_output_window.shape[0], real_output_window.shape[1], sampled_n))
            outlier_score_tmp = np.zeros((real_output_window.shape[0], sampled_n))
            for i in range(pred_window_mean.shape[0]):
                post_pred = stats.norm.rvs(pred_window_mean[i], pred_window_std[i], size=sampled_n)
                post_pred_tmp[i, :, :] = post_pred.reshape(1, -1)

            for i in range(outlier_score_tmp.shape[1]):
                # y_diff = np.concatenate([output_step_window.values, post_pred_tmp[:,:,i]], axis=1)
                y_diff = real_output_window - post_pred_tmp[:, :, i]
                y_diff[y_diff == -np.inf] = 1
                y_diff[y_diff == np.inf] = 1
                y_diff[np.isnan(y_diff)] = 1

                outlier_score = self.outlier_score(y_diff)
                outlier_score_tmp[:, i] = outlier_score
            outlier_score_mus = outlier_score_tmp.mean(axis=1)
            outlier_score_std = outlier_score_tmp.std(axis=1)
        else:
            outlier_score_mus = self.outlier_score(real_output_window - pred_window_mean)
            outlier_score_std = pred_window_std.mean(axis=1)  # outlier_score_tmp.std(axis=1)

        outlier_result = outlier_score_mus >= self.detect_model.threshold
        outlier_prob = 1 - norm.cdf(self.detect_model.threshold, loc=outlier_score_mus, scale=outlier_score_std)
        outlier_prob[np.isnan(outlier_prob)] = 1
        return outlier_prob

    def anomaly_detection(self, outlier_probability, outlier_score_input, smooth_parm, outlier_prob_threadhold=0.4,
                          outlier_score_input_threadhold=0.4):
        """
        outlier_probability: numpy.array with shape(N), N: len of data
        outlier_prob_threadhold: float, anomaly_prob_threadhold
        return:
        outlier_cluster/anomaly: dict(), start and end timepoint of novelty
        """
        anomaly = dict()
        anomaly['start'] = []
        anomaly['end'] = []
        smooth_outlier_score_prob = self.data_process.smooth(outlier_probability.values, 18)
        outlier_dates = outlier_probability[smooth_outlier_score_prob > outlier_prob_threadhold].index
        # outlier_dates = outlier_probability[outlier_probability.values>outlier_prob_threadhold].index
        if len(outlier_dates) == 0:
            return anomaly

        if outlier_score_input_threadhold > 0:
            outlier_input_dates = outlier_score_input[outlier_score_input > outlier_score_input_threadhold].index
            outlier_dates_filter = []
            for i in outlier_dates:
                if i in outlier_input_dates:
                    outlier_dates_filter.append(i)
            if len(outlier_dates_filter) == 0:
                return anomaly
        else:
            outlier_dates_filter = outlier_dates

        outlier_window_sec = (pd.Index(outlier_dates_filter).values.astype(float) // 10 ** 9).reshape(-1, 1)
        eps_sec = pd.Timedelta(minutes=10).total_seconds() * 36

        clustering = DBSCAN(eps=eps_sec, min_samples=18).fit(outlier_window_sec)
        anomaly_ids = np.unique(clustering.labels_[clustering.labels_ != -1])
        for anomaly_id in anomaly_ids:
            anomaly_start = pd.Index(outlier_dates_filter)[clustering.labels_ == anomaly_id].min()
            anomaly_end = pd.Index(outlier_dates_filter)[clustering.labels_ == anomaly_id].max()

            anomaly['start'].append(anomaly_start)
            anomaly['end'].append(anomaly_end)
        return anomaly


    #process data for DBSCAN
    def data_tranform(self, data, interpolation_v=-999, sep_v=-998):
        """
        data: numpy.array with shape(N, M, Q), N: num of series, M: length of series, Q: num of channel
        return:
        trans_data: numpy.array with shape(N, M*Q),
        """
        #In: shape (Series, length, Channel)
        #Out: shape (Series, length * Channel)
        trans_data = data.copy()
        where_are_NaNs = np.isnan(trans_data)
        trans_data[where_are_NaNs] = interpolation_v
        sep = np.zeros((data.shape[0], 1, data.shape[2])) + sep_v
        trans_data = np.hstack((trans_data, sep))
        trans_data = np.transpose(trans_data, (0, 2, 1)).reshape(data.shape[0], -1)
        return trans_data

    def data_retranform(self, trans_data, interpolation_v=-999, sep_v=-998):
        """
        trans_data: numpy.array with shape(N, M*Q), N: num of series, M: length of series, Q: num of channel
        return:
        data: numpy.array with shape(N, M, Q),
        """
        #In: shape (Series, length, Channel)
        #Out: shape (Series, length * Channel)
        data = trans_data.copy()
        series = trans_data.shape[0]
        num_channel = np.count_nonzero(data[0] == sep_v)
        length = int(trans_data.shape[1]/num_channel)
        data = data.reshape(series, num_channel, length)
        data[data == interpolation_v ] = np.nan
        data = np.transpose(data, (0, 2, 1))
        data = data[:,:-1,:]
        return data

    def dtw_distance(self, s1,s2):
        """
        s1: numpy.array with shape(1, M*Q),  M: length of series, Q: num of channel
        s2: numpy.array with shape(1, M*Q),  M: length of series, Q: num of channel
        return:
        distance: int,
        """
        s1_retrans = self.data_retranform(s1.reshape(1, -1))
        s2_retrans = self.data_retranform(s2.reshape(1, -1))
        return dtw(s1_retrans[0], s2_retrans[0])