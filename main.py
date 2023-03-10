import os
import torch
import torch.nn as nn
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.cluster import DBSCAN
from torchensemble.utils import io
from sklearn.preprocessing import MinMaxScaler
from utils.probabilistic_forecasting_models import train, HeteroscedasticDropoutNetwork, Heteroscedastic_lstm, \
    VotingRegressor_modified
from utils.plot import bokeh_data_clean, visualization, anomaly_clustering_plot, shap_explain, db_gen
from utils.simulation_system import SimulationSystem
from utils.preprocessing import DataPreprocess
from utils.utils import gaussian_nll_loss, dataset
from utils.outlier_detection_models import mahalanobis_model, isolationforest_model
import json
import time


# root_dir = '/mnt/stud/work/sgao/Gaussian_Dropout_DTS'
# root_dir = 'D:\IES\prob_forecast_yhe'
root_dir = os.getcwd()

def get_args_parser(add_help=True):
    import argparse
    parser = argparse.ArgumentParser(description='', add_help=add_help)
    parser.add_argument('--data-path', default=f'{root_dir}/real_dataset', help='dataset')
    parser.add_argument('--anomaly-info-path', default=f'{root_dir}/real_dataset/anomalies_timestamp.csv',
                        help='dataset')
    parser.add_argument('--path-warm-online-info', default=f'{root_dir}/anomaly_conf/warm_online_split.csv',
                        help='dataset')
    parser.add_argument('--exp_conf_path', default=f'{root_dir}/anomaly_conf/',
                        help='dataset')
    parser.add_argument('--num-inverter',
                            default="artificial_dataset",
                        help='number of inverter or artificial_dataset')
    parser.add_argument('--synthetic-anomaly-type',
                        default='m',
                        help='number of inverter')
    parser.add_argument('--train-methods',
                        default="MCDHDNN",
                        help='method of training (MCDHDNN, VEHDNN, VEHLSTM)')
    parser.add_argument('--visual-bokeh-path', default=f'{root_dir}/visualization',
                        help='bokeh format visualization path where to save')
    return parser


def create_path(args):
    ## create path of visualization
    os.makedirs(args.visual_bokeh_path, exist_ok=True)


def load_data(args, data_conf, synthetic_anomaly_confs=None):
    print("Load data and add synthetic anomalies")
    data_process = DataPreprocess()

    warm_online_split_info = pd.read_csv(args.path_warm_online_info, sep=";")
    warmup_start = \
        warm_online_split_info[warm_online_split_info['inverter'] == str(args.num_inverter)]['warm_start'].values[0]
    warmup_end = \
        warm_online_split_info[warm_online_split_info['inverter'] == str(args.num_inverter)]['warm_end'].values[0]
    online_start = \
        warm_online_split_info[warm_online_split_info['inverter'] == str(args.num_inverter)]['online_start'].values[0]
    online_end = \
        warm_online_split_info[warm_online_split_info['inverter'] == str(args.num_inverter)]['online_end'].values[0]

    if args.num_inverter == 'artificial_dataset':
        inverter_data_norm = data_process.load_artificial_data(data_conf, synthetic_anomaly_confs)
        inverter_data_warm = inverter_data_norm[data_conf['input_ch_name'] + data_conf['output_ch_name']][
                                  pd.Timestamp(warmup_start):pd.Timestamp(warmup_end)]
        inverter_data_online = inverter_data_norm[data_conf['input_ch_name'] + data_conf['output_ch_name']][
                                    pd.Timestamp(online_start):pd.Timestamp(online_end)]
        inverter_data_norm_warm, inverter_data_norm_online, scaler = data_process.normalization(
            inverter_data_warm,
            inverter_data_online,
            MinMaxScaler())
    else:
        '''
        # for real data
        inverter_df_numeric, inverter_df_category = data_process.load_inverter_data(args.path_dataset)
        inverter_warm = inverter_df_numeric[data_conf['input_ch_name'] + data_conf['output_ch_name']][
                        pd.Timestamp(warmup_start):pd.Timestamp(warmup_end)]
        inverter_online = inverter_df_numeric[data_conf['input_ch_name'] + data_conf['output_ch_name']][
                          pd.Timestamp(online_start):pd.Timestamp(online_end)]
        
        # Interpolation
        inverter_warm_clean, _, _, _ = data_process.interpolation(inverter_warm,
                                                                  data_conf['input_ch_name'],
                                                                  data_conf['output_ch_name'],
                                                                  data_conf['interpolation_days'],
                                                                  clean_input=True)
    
        # Normalization
        inverter_data_norm_warm, inverter_data_norm_online, scaler = data_process.normalization(
            inverter_warm_clean,
            inverter_online,
            MinMaxScaler())
        '''

        #for norm data
        '''
        inverter_data_norm = pd.concat([inverter_data_norm_warm, inverter_data_norm_online])
        inverter_data_norm.to_csv(f'{root_dir}/real_norm_dataset/10.csv')
        '''

        inverter_data_norm = data_process.load_inverter_norm_data(f'{args.path_dataset}')
        inverter_data_norm_warm = inverter_data_norm[data_conf['input_ch_name'] + data_conf['output_ch_name']][
                        pd.Timestamp(warmup_start):pd.Timestamp(warmup_end)]
        inverter_data_norm_online = inverter_data_norm[data_conf['input_ch_name'] + data_conf['output_ch_name']][
                          pd.Timestamp(online_start):pd.Timestamp(online_end)]

    if inverter_data_norm_warm.index[0] > inverter_data_norm_online.index[0]:
        inverter_data_norm = pd.concat([inverter_data_norm_online, inverter_data_norm_warm])
        start = online_start
        end = warmup_end
    elif inverter_data_norm_warm.index[0] < inverter_data_norm_online.index[0]:
        inverter_data_norm = pd.concat([inverter_data_norm_warm, inverter_data_norm_online])
        start = warmup_start
        end = online_end

    # Building warm-up phase sequence
    features_sequence, targets_sequence = data_process.build_time_sequence(inverter_data_norm_warm,
                                                                           data_conf['input_ch_name'],
                                                                           data_conf['output_ch_name'],
                                                                           nr_in_steps=data_conf[
                                                                               'history_input_length'],
                                                                           nr_out_steps=data_conf[
                                                                               'pred_ahead_output_length'],
                                                                           interval=data_conf['interval'],
                                                                           autoregression=False,
                                                                           resolution=data_conf['resolution'])

    features_warm = features_sequence
    latest_pred_each_channel = [(i + 1) * (data_conf['pred_ahead_output_length'] + 1) - 1 for i in
                                range(len(data_conf['output_ch_name']))]
    targets_warm = targets_sequence[targets_sequence.columns[latest_pred_each_channel]]

    # Building online phase sequence
    features_sequence, targets_sequence = data_process.build_time_sequence(inverter_data_norm_online,
                                                                           data_conf['input_ch_name'],
                                                                           data_conf['output_ch_name'],
                                                                           nr_in_steps=data_conf[
                                                                               'history_input_length'],
                                                                           nr_out_steps=data_conf[
                                                                               'pred_ahead_output_length'],
                                                                           interval=data_conf['interval'],
                                                                           autoregression=False,
                                                                           resolution=data_conf['resolution'])

    features_online = features_sequence
    targets_online = targets_sequence[targets_sequence.columns[latest_pred_each_channel]]

    ahead_time = pd.Timedelta(hours=data_conf['pred_ahead_output_length'])
    targets_online.index = targets_online.index + ahead_time
    rename_list = []
    for name in targets_online.keys():
        rename_list.append(name.split('+')[0])
    targets_online.columns = rename_list
    if args.train_methods == 'VEHLSTM':
        features_warm = features_warm.iloc[:, :-2]
        features_online = features_online.iloc[:, :-2]
    return features_warm, targets_warm, features_online, targets_online, inverter_data_norm \
        , start, end


def load_forecast_models(forecast_model_path, model, features_warm, targets_warm, model_conf, data_conf):
    print("Load forecasting models and train anomaly detection models")
    if args.train_methods == 'MCDHDNN':
        try:
            with open(f'{forecast_model_path}/MCDHDNN.pickle', 'rb') as f:
                GD_model = pickle.load(f)
                # simu_sys_warm = None
                print(f"Load forecasting models from path: {forecast_model_path}")
            return GD_model

        except FileNotFoundError or IsADirectoryError:
            GD_model = warm_up(args, features_warm, targets_warm, model_conf, data_conf)
            print(f"Load forecasting models through warm-up training")
            return GD_model

    elif args.train_methods == 'VEHDNN' or 'VEHLSTM':
        try:
            voting_model = VotingRegressor_modified(
                estimator=model,
                n_estimators=10,
                cuda=False,
            )
            io.load(voting_model, save_dir=forecast_model_path)
            print(f"Load forecasting models from path: {forecast_model_path}")
            return voting_model

        except FileNotFoundError:
            voting_model = warm_up(args, features_warm, targets_warm, model_conf, data_conf)
            print(f"Load forecasting models through warm-up training")
            return voting_model


def load_models(features_warm, targets_warm, forecast_model_path, detection_conf, model_conf, data_conf):
    if args.train_methods == 'VEHLSTM':
        lstm_model = Heteroscedastic_lstm(input_size=model_conf["input_size"], hidden_size=model_conf["hidden_size"]
                                          , num_layers=model_conf["num_layers"], output_size=model_conf["output_size"]
                                          , proj_size=model_conf["proj_size"])
        model = load_forecast_models(forecast_model_path, lstm_model, features_warm, targets_warm, model_conf,
                                     data_conf)
    else:
        GD_model = HeteroscedasticDropoutNetwork(input_size=features_warm.shape[1],
                                                 ann_structure=model_conf["ann_structure"],
                                                 output_size=targets_warm.shape[1] + 1,
                                                 activation_f=nn.LeakyReLU(0.05),
                                                 dropout_method=model_conf["dropout_method"],
                                                 dropout_rate=model_conf["dropout_rate"],
                                                 alpha=None,
                                                 beta=None,
                                                 weight_drop=0
                                                 )
        model = load_forecast_models(forecast_model_path, GD_model, features_warm, targets_warm, model_conf, data_conf)

    X_warm_torch, y_warm_torch = torch.Tensor(features_warm.values), torch.Tensor(targets_warm.values)

    if args.train_methods == 'VEHLSTM':
        X_warm_torch = X_warm_torch.reshape(-1, data_conf["history_input_length"], model_conf["input_size"])

    predictions_warm = model.predict_prob(X_warm_torch, iters=model_conf["sample_num"])

    # Yt_hat_warm = predictions_warm[0]
    # sampled_mus_warm = predictions_warm[1]
    mean_mus_warm = predictions_warm[2]
    # noises_warm = predictions_warm[3]
    # aleatoric_warm = predictions_warm[4]
    # epistemic_warm = predictions_warm[5]

    mahalanobis_detect_model = mahalanobis_model(ppf=detection_conf['detect_model_threshold'])
    error_train = (y_warm_torch - mean_mus_warm.reshape_as(y_warm_torch)).numpy()
    mahalanobis_detect_model.fit(error_train)
    score = mahalanobis_detect_model.outlier_score(error_train)
    # print(score)
    print("mahalanobis threshold: ", mahalanobis_detect_model.threshold)

    if_detect_model = isolationforest_model(ppf=detection_conf['detect_model_threshold'])
    if_error_train = (y_warm_torch - mean_mus_warm.reshape_as(y_warm_torch)).numpy()
    if_detect_model.fit(error_train)
    if_score = if_detect_model.outlier_score(if_error_train)
    # print(if_score)
    print("isolation forest threshold: ", if_detect_model.threshold)

    return model, mahalanobis_detect_model, if_detect_model


def warm_up(args, features_warm, targets_warm, model_conf, data_conf):
    # Training/validation dataset
    X_train_torch, y_train_torch, X_val_torch, y_val_torch, dataloader = dataset(features=features_warm,
                                                                                 targets=targets_warm,
                                                                                 val_size=data_conf["val_size"],
                                                                                 batch_size=data_conf["batch_size"],
                                                                                 method=args.train_methods)

    print(f'Start warm-up training inverter: {args.num_inverter}')
    criterion = gaussian_nll_loss
    if args.train_methods == 'VEHLSTM':
        model = Heteroscedastic_lstm(input_size=model_conf["input_size"], hidden_size=model_conf["hidden_size"]
                                     , num_layers=model_conf["num_layers"], output_size=model_conf["output_size"]
                                     , proj_size=model_conf["proj_size"])
    else:
        model = HeteroscedasticDropoutNetwork(input_size=features_warm.shape[1],
                                              ann_structure=model_conf["ann_structure"],
                                              output_size=targets_warm.shape[1] + 1,
                                              activation_f=nn.LeakyReLU(0.05),
                                              dropout_method=model_conf["dropout_method"],
                                              dropout_rate=model_conf["dropout_rate"],
                                              alpha=None,
                                              beta=None,
                                              weight_drop=0
                                              )

    if args.train_methods == 'VEHDNN' or args.train_methods == 'VELSTM':
        model = VotingRegressor_modified(
            estimator=model,
            n_estimators=model_conf['sample_num'],
            cuda=False,
        )
        model.set_criterion(criterion)

        model.set_optimizer('Adam',  # parameter optimizer
                            lr=model_conf['lr'],  # learning rate of the optimizer
                            weight_decay=model_conf['weight_decay'])  # weight decay of the optimize

        # Training
        model.fit(train_loader=dataloader,  # training data
                  epochs=model_conf['epochs'],
                  save_dir=args.forecast_model_path)  # the number of training epochs
        return model

    elif args.train_methods == 'MCDHDNN':
        optimizer = torch.optim.Adam(model.parameters(), lr=model_conf['lr'], weight_decay=model_conf['weight_decay'])
        model = train(model=model,
              criterion=criterion,
              optimizer=optimizer,
              dataloader=dataloader,
              epochs=model_conf['epochs'],
              X_train_torch=X_train_torch,
              y_train_torch=y_train_torch,
              X_val_torch=X_val_torch,
              y_val_torch=y_val_torch,
              plot_path="./warmup_models/",
              phase='warm-up',
              serial_number=str(args.num_inverter))

        # save the trained model
        warm_output_dir = os.path.join(args.forecast_model_path,
                                       'MCDHDNN')
        warm_up_path_pickle = warm_output_dir + '.pickle'
        with open(warm_up_path_pickle, 'wb') as f:
            pickle.dump(model, f)
        return model


def warmup_data_prediction(pred_model, input_signals, output_signals, method, model_conf, data_conf):
    output_keys = output_signals.keys()
    if method == 'VEHLSTM':
        input_signals_v = input_signals.values.reshape(-1, data_conf['history_input_length'], model_conf["input_size"])
    else:
        input_signals_v = input_signals.values
    predictions = pred_model.predict_prob(torch.tensor(input_signals_v).float(), iters=model_conf["sample_num"])
    Yt_hat_pred = predictions[0]
    mus = torch.mean(Yt_hat_pred[:, :int(Yt_hat_pred.shape[1] / 2), :], dim=-1).numpy()
    sigmas = torch.mean(torch.exp(Yt_hat_pred[:, int(Yt_hat_pred.shape[1] / 2):, :]), dim=-1).numpy()
    pred_mus = pd.DataFrame(mus, index=input_signals.index, columns=output_keys)
    pred_std = pd.DataFrame(sigmas, index=input_signals.index, columns=output_keys)
    return pred_mus, pred_std

def experiment_conf(args):
    try:
        f = open(f"{args.exp_conf_path}anomaly_info_{args.num_inverter}_{args.synthetic_anomaly_type}.json")  
    except:
        f = open(f"{args.exp_conf_path}anomaly_info_{args.num_inverter}.json")
    confs = json.load(f)
    detection_conf = confs['detection_conf'][args.train_methods]

    data_conf = confs['data_conf']
    model_conf = confs['model_conf'][args.train_methods]
    if args.num_inverter == 'artificial_dataset':
        anomaly_conf = confs['anomaly_info']
    else:
        anomaly_conf = []
        failure_df = pd.read_csv(args.anomaly_info_path, sep=';')
        anomaly_start_time = failure_df[failure_df['Series_Number'] == int(args.num_inverter)]['Anomaly_Start_Time']
        anomaly_end_time = failure_df[failure_df['Series_Number'] == int(args.num_inverter)]['Anomaly_End_Time']
        for i in range(len(anomaly_end_time)):
            anomaly_conf_item = dict()
            anomaly_conf_item['start'] = anomaly_start_time.values[i]
            anomaly_conf_item['end'] = anomaly_end_time.values[i]
            anomaly_conf_item['type'] = "real"
            anomaly_conf.append(anomaly_conf_item)
    return detection_conf, anomaly_conf, data_conf, model_conf

def online(args):
    st = time.time()
    # create path
    create_path(args)

    # load experiment_conf
    detection_conf, anomaly_conf, data_conf, model_conf = experiment_conf(args)

    # load data
    features_warm, targets_warm, features_online, targets_online, inverter_data_norm \
        , start, end, = load_data(args, data_conf, anomaly_conf)

    plt.figure(figsize=(16, 6))
    plt.plot(inverter_data_norm.index, inverter_data_norm.values[:,:5], alpha=0.1)
    plt.plot(inverter_data_norm.index, inverter_data_norm.values[:,5:], alpha=0.5)
    plt.ylim(-1,2.5)
    plt.show()

    ## get the model of warm-up phase and online data
    print(f'forecast_model_path: {args.forecast_model_path}')
    model, mahalanobis_detect_model, if_detect_model = load_models(features_warm, targets_warm,
                                                                   args.forecast_model_path, detection_conf, model_conf,
                                                                   data_conf)
    detection_name = detection_conf["detection_model"].lower()
    if detection_name == "mahalanobis":
        detect_model = mahalanobis_detect_model
    elif detection_name == "isolationforest":
        detect_model = if_detect_model
    else:
        raise RuntimeError(
            f"Invalid detection model {args.detection_model}. Only mahalanobis, isolationforest are supported.")

    if_detect_model_input_checker = isolationforest_model(n_estimators=100,
                                                          ppf=0.99) #detection_conf['detect_model_threshold']
    if_detect_model_input_checker.fit(features_warm)
    if_score_input = if_detect_model_input_checker.outlier_score(features_online)
    if_score_input_checker = pd.DataFrame(if_score_input
                                          , index=features_online.index-pd.Timedelta(hours=data_conf['pred_ahead_output_length']/2)
                                          , columns=["outlier_score_input"])
    if detection_conf["is_input_check"] is True:
        plt.plot(if_score_input_checker)
        plt.show()
        print("isolation forest input_checker threshold: ", if_detect_model_input_checker.threshold)
        plt.plot(if_detect_model_input_checker.outlier_score(features_online))

        # plt.show()
        plt.savefig(args.visual_bokeh_path + '/' + 'outlier_score_input_online.png')
        plt.show()
    et = time.time()
    elapsed_time = et - st
    print(' data model prepare Execution time:', elapsed_time, 'seconds')

    st = time.time()
    warm_up_pred_mus, warm_up_pred_std = warmup_data_prediction(pred_model=model, input_signals=features_warm,
                                                                output_signals=targets_warm
                                                                , method=args.train_methods, model_conf=model_conf,
                                                                data_conf=data_conf)
    et = time.time()
    elapsed_time = et - st
    print(' simu_warm Execution time:', elapsed_time, 'seconds')

    st = time.time()
    simu_sys = SimulationSystem(pred_model=model, detect_model=detect_model,
                                if_score_input_checker=if_score_input_checker,
                                input_signals=features_online,
                                output_signals=targets_online,
                                pred_window_size=pd.Timedelta(hours=data_conf["pred_ahead_output_length"]),
                                history_window_size=pd.Timedelta(days=10), real_failure_list=None)
    if detection_conf['is_input_check'] is True:
        deviation_score_input_threadhold = if_detect_model_input_checker.threshold
    else:
        deviation_score_input_threadhold = 0
    simu_sys.simulation(smooth_parm=data_conf["outlier_prob_smooth"],
                        outlier_prob_threadhold=detection_conf['outlier_prob_threshold']
                        , deviation_score_input_threadhold=deviation_score_input_threadhold
                        , method=args.train_methods, mode="online")

    et = time.time()
    elapsed_time = et - st
    print(' simu_sys Execution time:', elapsed_time, 'seconds')

    anomaly_cluster = simu_sys.get_anomaly_cluster(eps=detection_conf['anomaly_clustering_eps'])

    anomaly_plot = pd.DataFrame({'anomalies_start': simu_sys.anomaly_time['start']
                                    , 'anomalies_end': simu_sys.anomaly_time['end']
                                    , 'cluster_result': anomaly_cluster.labels_})

    anomaly_list = simu_sys.get_anomaly_list()

    cluster2anomaly = dict()
    cluster2anomaly[-1] = -1
    cluster2anomaly[0] = 1
    cluster2anomaly[1] = 2
    cluster2anomaly[2] = 4
    cluster2anomaly[3] = 3

    # 在这里用于做人工异常聚类
    if detection_conf['is_clustering'] == False:
        cluster2anomaly = None

    anomaly_clustering_plot(anomaly_plot, anomaly_list, cluster2anomaly=cluster2anomaly)
    
    shap_explain(simu_sys, inverter_data_norm, data_conf['input_ch_name'], data_conf['output_ch_name']
                 , time_start="2019-02-28 16:00", time_end="2019-03-02 00:00"
                 , time_middle="2019-03-01 08:00", smooth_parm=data_conf['outlier_prob_smooth']
                 , save_path=args.visual_bokeh_path)
    
    simu_sys.auc_anomaly_evaluation(anomaly_conf, cluster_confs=anomaly_plot, cluster2anomaly=cluster2anomaly,
                                    smooth_par=data_conf["outlier_prob_smooth"], warmup_len=len(features_warm))

    print("Save the visualization to html.")
    #db_gen(simu_sys, anomaly_cluster, warm_up_pred_mus, inverter_data_norm, str(args.num_inverter),
    #              data_conf["input_ch_name"],data_conf["output_ch_name"],
    #              start, end, [], args.visual_bokeh_path,
    #              smooth_par=detection_conf["outlier_prob_smooth"], method=args.train_methods,
    #              synthetic_anomaly_type=args.synthetic_anomaly_type)

    visualization(simu_sys, anomaly_cluster, warm_up_pred_mus, inverter_data_norm, str(args.num_inverter),
                  data_conf["input_ch_name"],data_conf["output_ch_name"],
                  start, end, [], args.visual_bokeh_path,
                  smooth_par=detection_conf["outlier_prob_smooth"], method=args.train_methods,
                  synthetic_anomaly_type=args.synthetic_anomaly_type)
    print(f'bokeh visualization are saved in {args.visual_bokeh_path}')

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    if args.num_inverter == 'artificial_dataset':
        args.path_dataset = None
    args.path_dataset = args.data_path + '/' + str(args.num_inverter) + '.csv'
    args.forecast_model_path = f'{root_dir}/warmup_models/inverter_{args.num_inverter}/'
    print(f'train_method: {args.train_methods}')
    print(f'forecast_model_path_main: {args.forecast_model_path}')
    online(args)
