import pandas as pd
import warnings
import numpy as np
import matplotlib.pyplot as plt
import json
warnings.filterwarnings("ignore")
from .alarm_system import AlarmSystem
from .detection_system import DetectionSystem
from sklearn.cluster import DBSCAN
from sklearn import metrics
from .preprocessing import DataPreprocess
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import multilabel_confusion_matrix
from utils.plot import plot_confusion_matrix
class SimulationSystem:
    def __init__(self, pred_model, detect_model, if_score_input_checker, input_signals, output_signals, pred_window_size=pd.Timedelta(days=1),
                 history_window_size=pd.Timedelta(days=10), real_failure_list=None):
        """
        pred_model: object, probabilistic forecasting models
        detect_model: object, outlier detection models
        pred_window_size: pd.Timedelta, windows size of prediction, for example next 1 days data
        history_window_size: pd.Timedelta, windows size for analysis historical data, e.g.calibration_score,
        real_failure_list: pd.Dataframe, real_failure_list
        """
        self.alarm_sys = AlarmSystem(pred_model, pred_window_size, history_window_size)
        self.detect_sys = DetectionSystem(pred_model, detect_model, pred_window_size, real_failure_list)
        #self.detect_sys_input = DetectionSystem(pred_model, detect_model_input, pred_window_size, real_failure_list)
        self.real_input_signals = input_signals
        self.real_output_signals = output_signals
        self.output_keys = output_signals.keys()
        self.signal_timestamp = input_signals.index
        self.pred_window_size = pred_window_size
        self.history_window_size = history_window_size
        self.real_failure_list = real_failure_list
        self.pred_output_signals_mus = None
        self.pred_output_signals_std = None
        self.pred_output_signals_aleatoric = None
        self.pred_output_signals_epistemic = None
        self.outlier_info_list = None
        self.crp_score_list = None
        self.QUANTILES = self.alarm_sys.create_quantiles(11, min_q=0.01, max_q=0.99)
        self.calibration_date_list = []
        self.calibration_score_list = []
        self.anomaly_time = dict()
        self.anomaly_time['start'] = []
        self.anomaly_time['end'] = []
        self.outlier_score_input = if_score_input_checker

    def get_anomaly_list(self):
        anomaly_df = pd.concat([self.pred_output_signals_mus.copy(), self.real_output_signals], axis=1, join='inner')
        max_len_index = np.argmax(np.array(self.anomaly_time['end']) - np.array(self.anomaly_time['start']))
        max_ts_len = len(
            anomaly_df[self.anomaly_time['start'][max_len_index]:self.anomaly_time['end'][max_len_index]])

        anomaly_list = np.full([len(self.anomaly_time['start']), max_ts_len, len(anomaly_df.keys())], np.nan)
        for anomaly_id in range(len(self.anomaly_time['start'])):
            ts = anomaly_df[
                 self.anomaly_time['start'][anomaly_id]: self.anomaly_time['end'][anomaly_id]].values
            anomaly_list[anomaly_id][0:len(ts)] = ts
        return anomaly_list

    def get_anomaly_cluster(self, eps):
        dbscan = DBSCAN(eps=eps, min_samples=5, metric=lambda a, b: self.detect_sys.dtw_distance(a, b))
        anomaly_list = self.get_anomaly_list()
        anomaly_clustering = dbscan.fit(self.detect_sys.data_tranform(anomaly_list))
        return anomaly_clustering

    # get post prediction for prediction window based on current window
    def post_pred_windows(self, curr_start_time, curr_end_time, pred_start_time, pred_end_time, method):
        """
        curr_start_time: pd.Timedate, the start timepoint of current window
        curr_end_time: pd.Timedate, the end timepoint of current window
        pred_start_time: pd.Timedate, the start timepoint of prediction window
        pred_end_time: pd.Timedate, the end timepoint of prediction window
        """
        curr_input_window = self.real_input_signals[curr_start_time:curr_end_time][:-1]
        # get pred_output_window
        # append it to self.pred_output_signals
        if method == 'VEHLSTM':
            curr_input_window = curr_input_window.values.reshape(-1, 24, 5)
            Yt_hat_pred, y_pred, aleatoric_pred, epistemic_pred = self.alarm_sys.pred_win_target(
                curr_input_window)
        else:
            Yt_hat_pred, y_pred, aleatoric_pred, epistemic_pred = self.alarm_sys.pred_win_target(curr_input_window.values)
        mus, sigmas = self.alarm_sys.postprocess_samples(Yt_hat_pred)
        pred_tmp_mus = pd.DataFrame(mus
                                    , index=self.real_output_signals[pred_start_time:pred_end_time][:-1].index
                                    , columns=self.output_keys)
        pred_tmp_std = pd.DataFrame(sigmas
                                    , index=self.real_output_signals[pred_start_time:pred_end_time][:-1].index
                                    , columns=self.output_keys)
        pred_tmp_aleatoric_pred = pd.DataFrame(aleatoric_pred.detach().numpy()
                                               ,
                                               index=self.real_output_signals[pred_start_time:pred_end_time][:-1].index
                                               , columns=self.output_keys)
        pred_tmp_epistemic_pred = pd.DataFrame(epistemic_pred.detach().numpy()
                                               ,
                                               index=self.real_output_signals[pred_start_time:pred_end_time][:-1].index
                                               , columns=self.output_keys)

        if self.pred_output_signals_mus is None:
            self.pred_output_signals_mus = pred_tmp_mus
            self.pred_output_signals_std = pred_tmp_std
            self.pred_output_signals_aleatoric = pred_tmp_aleatoric_pred
            self.pred_output_signals_epistemic = pred_tmp_epistemic_pred
        else:
            self.pred_output_signals_mus = pd.concat([self.pred_output_signals_mus, pred_tmp_mus])
            self.pred_output_signals_std = pd.concat([self.pred_output_signals_std, pred_tmp_std])
            self.pred_output_signals_aleatoric = pd.concat(
                [self.pred_output_signals_aleatoric, pred_tmp_aleatoric_pred])
            self.pred_output_signals_epistemic = pd.concat(
                [self.pred_output_signals_epistemic, pred_tmp_epistemic_pred])

    # when we get the real target for the current window, then we can calculate outlier probability
    def outlier_prob_windows(self, curr_start_time, curr_end_time, outlier_prob_threadhold, deviation_score_input_threadhold, smooth_parm, outlier_score_input_tmp):
        """
        curr_start_time: pd.Timedate, the start timepoint of current window
        curr_end_time: pd.Timedate, the end timepoint of current window
        """
        # outlier detection, novelty detection
        # append it to self.outlier_prob
        curr_output_window = self.real_output_signals[curr_start_time:curr_end_time][:-1]

        pred_output_window_mus = self.pred_output_signals_mus[curr_start_time:curr_end_time][:-1]
        # pred_output_window_std = self.pred_output_signals_std[curr_start_time:curr_end_time][:-1]
        pred_output_window_std = ((self.pred_output_signals_aleatoric[curr_start_time:curr_end_time][:-1] ** 2) +
                                  (self.pred_output_signals_epistemic[curr_start_time:curr_end_time][:-1] ** 2)) ** 0.5
        outlier_scores = self.detect_sys.outlier_score(
            (curr_output_window.values - pred_output_window_mus.values)).reshape(-1, 1)
        outlier_probs = self.detect_sys.outlier_prob(curr_output_window.values, pred_output_window_mus.values,
                                                     pred_output_window_std.values).reshape(-1, 1)

        outlier_info_tmp = pd.DataFrame(np.concatenate((outlier_scores, outlier_probs), axis=1)
                                        , index=curr_output_window.index
                                        , columns=["outlier_score", "outlier_prob"])

        if self.outlier_info_list is None:
            self.outlier_info_list = outlier_info_tmp
        else:
            self.outlier_info_list = pd.concat([self.outlier_info_list, outlier_info_tmp])
        outlier_cluster = self.detect_sys.anomaly_detection(outlier_info_tmp["outlier_prob"], outlier_score_input_tmp["outlier_score_input"], smooth_parm, outlier_prob_threadhold, deviation_score_input_threadhold)
        self.anomaly_time['start'].extend(outlier_cluster['start'])
        self.anomaly_time['end'].extend(outlier_cluster['end'])

    # Analysis historic window, for example calibration score, crp score
    def history_analysis_windows(self, curr_start_time, curr_end_time, history_start_time, history_end_time):
        """
        curr_start_time: pd.Timedate, the start timepoint of current window
        curr_end_time: pd.Timedate, the end timepoint of current window
        history_start_time: pd.Timedate, the start timepoint of history window
        history_end_time: pd.Timedate, the end timepoint of history window
        """
        # history data analysis, crp, colaberation score..
        # pred next n steps anomaly prob
        curr_output_window = self.real_output_signals[curr_start_time:curr_end_time][:-1]
        pred_output_window_mus = self.pred_output_signals_mus[curr_start_time:curr_end_time][:-1]
        pred_output_window_std = self.pred_output_signals_std[curr_start_time:curr_end_time][:-1]
        qpreds = self.alarm_sys.quantiles_preds(pred_output_window_mus.values, pred_output_window_std.values,
                                                self.QUANTILES)
        crp_score = self.alarm_sys.crp_score(curr_output_window, qpreds).mean(axis=1)
        # if(crp_score.mean()>0.8):
        #    print(curr_start_time)
        #    print(pred_output_window_std)
        crp_score_tmp = pd.DataFrame(crp_score
                                     , index=curr_output_window.index
                                     , columns=["crp_score"])

        if self.crp_score_list is None:
            self.crp_score_list = crp_score_tmp
        else:
            self.crp_score_list = pd.concat([self.crp_score_list, crp_score_tmp])

        history_output_window = self.real_output_signals[history_start_time:history_end_time][:-1]
        history_pred_window_mus = self.pred_output_signals_mus[history_start_time:history_end_time][:-1]
        history_pred_window_std = self.pred_output_signals_std[history_start_time:history_end_time][:-1]
        qpreds = self.alarm_sys.quantiles_preds(history_pred_window_mus.values, history_pred_window_std.values,
                                                self.QUANTILES)
        quantile_scores, calibration_score = self.alarm_sys.calibration_score(history_output_window.values, qpreds,
                                                                              self.QUANTILES)
        # print(curr_start_time, quantile_scores, calibration_score)
        self.calibration_date_list.append(curr_end_time)
        self.calibration_score_list.append(calibration_score)

    # Simulate to run alarm and detection system
    def simulation(self, smooth_parm, method, outlier_prob_threadhold=0.4, deviation_score_input_threadhold=0.4, mode="warm"):
        curr_start_time = self.signal_timestamp[0]
        curr_end_time = self.signal_timestamp[0] + self.pred_window_size

        pred_start_time = curr_start_time + self.pred_window_size
        pred_end_time = curr_end_time + self.pred_window_size

        history_start_time = curr_start_time
        history_end_time = curr_end_time

        while 1:
            self.post_pred_windows(curr_start_time, curr_end_time, pred_start_time, pred_end_time, method)
            if curr_start_time != self.signal_timestamp[0]:
                self.outlier_prob_windows(curr_start_time, curr_end_time, outlier_prob_threadhold, deviation_score_input_threadhold, smooth_parm, self.outlier_score_input)
                self.history_analysis_windows(curr_start_time, curr_end_time, history_start_time, history_end_time)

            curr_start_time = curr_end_time
            curr_end_time = curr_start_time + self.pred_window_size
            pred_start_time = curr_start_time + self.pred_window_size
            pred_end_time = curr_end_time + self.pred_window_size

            history_start_time = curr_start_time - self.history_window_size
            if history_start_time <= self.signal_timestamp[0]:
                history_start_time = self.signal_timestamp[0] + self.pred_window_size
            history_end_time = curr_end_time
            if pred_end_time >= self.signal_timestamp[-1]:
                break

    def outlier_prob_mask(self, anomaly_confs):
        tmp_outlier_prob = self.outlier_info_list.copy()
        tmp_outlier_prob.insert(2, "tmp", np.random.uniform(0, 0.05, len(tmp_outlier_prob)), True)
        for i in range(len(anomaly_confs)):
            anomaly_start = pd.to_datetime(anomaly_confs[i]['start'])
            anomaly_end = pd.to_datetime(anomaly_confs[i]['end'])
            start_time = anomaly_start - pd.Timedelta(hours=1)
            end_time = anomaly_end + pd.Timedelta(hours=1)
            tmp_outlier_prob['tmp'][start_time.strftime("%Y-%m-%d %H:%M"):end_time.strftime("%Y-%m-%d %H:%M")] \
                = self.outlier_info_list['outlier_prob'][start_time.strftime("%Y-%m-%d %H:%M"):end_time.strftime("%Y-%m-%d %H:%M")]
        self.outlier_info_list['outlier_prob'] = tmp_outlier_prob['tmp']

        novelty_id_match = []
        for nove_id in range(len(self.anomaly_time['start'])):
            nove_start = self.anomaly_time['start'][nove_id].replace(tzinfo=None)
            nove_end = self.anomaly_time['end'][nove_id].replace(tzinfo=None)
            for i in range(len(anomaly_confs)):
                anomaly_start = pd.to_datetime(anomaly_confs[i]['start'])
                anomaly_end = pd.to_datetime(anomaly_confs[i]['end'])
                if (nove_start <= anomaly_start and nove_end >= anomaly_start) or \
                   (nove_start >= anomaly_start and nove_start <= anomaly_end) or \
                   (nove_start <= anomaly_start and nove_end >= anomaly_end)or \
                   (nove_start >= anomaly_start and nove_end <= anomaly_end):
                    novelty_id_match.append(nove_id)
                    break
        self.anomaly_time['start'] = np.array(self.anomaly_time['start'])[novelty_id_match].tolist()
        self.anomaly_time['end'] = np.array(self.anomaly_time['end'])[novelty_id_match].tolist()

    def auc_anomaly_evaluation(self, anomaly_confs, cluster_confs, cluster2anomaly, smooth_par, warmup_len):

        start_time = self.signal_timestamp[0]
        end_time = self.signal_timestamp[-1]

        data_process = DataPreprocess()
        smooth_outlier_prob = data_process.smooth(self.outlier_info_list['outlier_prob'], smooth_par)
        outlier_prob = self.outlier_info_list.copy()
        outlier_prob['outlier_prob'] = smooth_outlier_prob

        outlier_prob = outlier_prob[start_time.strftime("%Y-%m-%d %H:%M"):end_time.strftime("%Y-%m-%d %H:%M")]

        #outlier_prob = self.outlier_info_list[start_time.strftime("%Y-%m-%d %H:%M"):end_time.strftime("%Y-%m-%d %H:%M")]
        outlier_prob.insert(2, "label", np.zeros(len(outlier_prob)), True)
        outlier_prob.insert(2, "pred_label", np.zeros(len(outlier_prob)), True)
        if anomaly_confs != None:
            for i in range(len(anomaly_confs)):
                outlier_prob['label'][anomaly_confs[i]['start']:anomaly_confs[i]['end']] = 1
            for i in range(len(cluster_confs)):
                outlier_prob['pred_label'][cluster_confs['anomalies_start'].iloc[i]:cluster_confs['anomalies_end'].iloc[i]] = 1
        fpr, tpr, thresholds = metrics.roc_curve(outlier_prob['label'].values
                                                 , outlier_prob['outlier_prob'].values, pos_label=1)
        auc_score = metrics.auc(fpr, tpr)
        #print(thresholds)
        plt.figure()
        lw = 2
        plt.plot(
            fpr,
            tpr,
            color="darkorange",
            lw=lw,
            label="ROC curve (area = %0.2f)" % auc_score,
        )
        plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("Receiver operating characteristic example")
        plt.legend(loc="lower right")
        plt.show()

        detect_lag = []
        for i in range(len(anomaly_confs)):
            real_anomaly_start = pd.to_datetime(anomaly_confs[i]['start'], utc=True)
            start_time_range_start = real_anomaly_start - pd.Timedelta(days=1)
            start_time_range_end = real_anomaly_start + pd.Timedelta(days=1)
            clost_detect_anomaly_start = 0
            close_diff = pd.Timedelta(days=20)
            close_symbol =1
            symbol = -1
            for j in range(len(cluster_confs)):
                detect_anomly_start = pd.to_datetime(cluster_confs['anomalies_start'].iloc[j], utc=True)
                if real_anomaly_start > detect_anomly_start:
                    diff = real_anomaly_start - detect_anomly_start
                    symbol = -1
                if real_anomaly_start <= detect_anomly_start:
                    diff = detect_anomly_start - real_anomaly_start
                    symbol = 1
                if diff < close_diff:
                    close_diff = diff
                    clost_detect_anomaly_start = detect_anomly_start
                    close_symbol = symbol
            if close_diff < pd.Timedelta(days=1):
                detect_lag.append(close_symbol * int(close_diff.total_seconds() / 60))
            print(clost_detect_anomaly_start, close_symbol, int(close_diff.total_seconds() / 60))




        f1_score_bin =f1_score(outlier_prob['label'].values, outlier_prob['pred_label'].values).ravel()
        f1_score_weighted = f1_score(outlier_prob['label'].values
                                     , outlier_prob['pred_label'].values
                                     , average='weighted').ravel()
        tn, fp, fn, tp = confusion_matrix(outlier_prob['label'].values, outlier_prob['pred_label'].values).ravel()
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        outlier_point_count = outlier_prob['label'].value_counts()[1.0]
        # outlier_point_count = outlier_prob['label'].value_counts()[0.0]
        all_point_count = len(self.outlier_info_list)
        outlier_rate_online = outlier_point_count/all_point_count
        outlier_rate_all = outlier_point_count / (all_point_count+ warmup_len)
        result = dict()
        result['auc_score']=auc_score
        result['outlier_point_count'] = outlier_point_count
        result['all_point_count_online'] = all_point_count
        result['all_point_count_warmup'] = warmup_len
        result['all_point_count_all'] = all_point_count + warmup_len
        result['outlier_rate_online'] = outlier_rate_online
        result['outlier_rate_all'] = outlier_rate_all
        result['f1_score_bin'] = f1_score_bin
        result['f1_score_weighted'] = f1_score_weighted
        result['precision'] = precision
        result['recall'] = recall
        result['tn'] = tn
        result['fp'] = fp
        result['fn'] = fn
        result['tp'] = tp
        if len(detect_lag)== 0:
            result['detect_lag_mean'] = 1
        else:
            result['detect_lag_mean'] = sum(detect_lag)/len(detect_lag)
        if cluster2anomaly is not None:
            outlier_prob.insert(2, "true_class", np.zeros(len(outlier_prob)), True)
            outlier_prob.insert(2, "cluster_class", np.zeros(len(outlier_prob)), True)
            for i in range(len(anomaly_confs)):
                outlier_prob['true_class'][anomaly_confs[i]['start']:anomaly_confs[i]['end']] = anomaly_confs[i]['anomaly_type']
            for i in range(len(cluster_confs)):
                outlier_prob['cluster_class'][cluster_confs['anomalies_start'].iloc[i]:cluster_confs['anomalies_end'].iloc[i]] = cluster2anomaly[cluster_confs['cluster_result'].iloc[i]]

            y_unique = outlier_prob['true_class'].unique()
            y_unique.sort()
            cm = confusion_matrix(outlier_prob['true_class'].values, outlier_prob['cluster_class'].values, labels=y_unique)
            plot_confusion_matrix(cm, ['Normal', 'A_1', 'A_2', 'A_3', 'A_4'], normalize=False)
            f1_score_cluster = f1_score(outlier_prob['true_class'].values
                                         , outlier_prob['cluster_class'].values
                                         , average='weighted').ravel()
            print(cm, f1_score_cluster)
        print(result)
        return result
