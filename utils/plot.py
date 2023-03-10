import torch

from .preprocessing import DataPreprocess
from bokeh.plotting import figure, output_file, save, show
from bokeh.models import Range1d
from bokeh.models import Span
from bokeh.layouts import column, row
from bokeh.models import PolyAnnotation, BoxAnnotation
from bokeh.models import CheckboxGroup, CustomJS
from bokeh.palettes import Dark2_5 as palette
import torch.nn as nn
import itertools
import xgboost
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import sqlite3

def bokeh_data_clean (inverter_warm_clean, inverter_warm, input_ch_name, output_ch_name, num_inverter, path_dataset):

    p = figure(title="Detection System", plot_width=1000, plot_height=300, x_axis_type='datetime')

    line_list = []
    p2 = figure(title="After Clear", plot_width=1000, plot_height=300, x_range=p.x_range, y_range=p.y_range,
                x_axis_type='datetime')
    colors = itertools.cycle(palette)

    for m, color in zip(input_ch_name, colors):
        # l = p2.line(inverter_data_norm_online.index, inverter_data_norm_online[m], alpha=0.5, line_color=color, legend_label=m)
        l = p2.line(inverter_warm_clean.index, inverter_warm_clean[m], alpha=0.5, line_color=color, legend_label=m)
        line_list.append(l)

    for m, color in zip(output_ch_name, colors):
        # l = p2.line(inverter_data_norm_online.index, inverter_data_norm_online[m], alpha=0.5, line_color=color, legend_label=m)
        l = p2.line(inverter_warm_clean.index, inverter_warm_clean[m], alpha=0.5, line_color=color, legend_label=m)
        line_list.append(l)

    checkbox2 = CheckboxGroup(labels=input_ch_name + output_ch_name, active=[0, 1, 2, 3, 4, 5], width=100)

    callback2 = CustomJS(
        args=dict(l0=line_list[0], l1=line_list[1], l2=line_list[2], l3=line_list[3], l4=line_list[4], l5=line_list[5],
                  checkbox=checkbox2), code="""
    l0.visible = checkbox.active.includes(0);
    l1.visible = checkbox.active.includes(1);
    l2.visible = checkbox.active.includes(2);
    l3.visible = checkbox.active.includes(3);
    l4.visible = checkbox.active.includes(4);
    l5.visible = checkbox.active.includes(5);
    """)
    checkbox2.js_on_change('active', callback2)
    p2.legend.label_text_font_size = '10pt'
    p2.add_layout(p2.legend[0], 'right')

    line_list = []
    p1 = figure(title="Original Sig", plot_width=1000, plot_height=300, x_range=p2.x_range, y_range=p2.y_range,
                x_axis_type='datetime')
    colors = itertools.cycle(palette)

    for m, color in zip(input_ch_name, colors):
        # l = p2.line(inverter_data_norm_online.index, inverter_data_norm_online[m], alpha=0.5, line_color=color, legend_label=m)
        l = p1.line(inverter_warm.index, inverter_warm[m], alpha=0.5, line_color=color, legend_label=m)
        line_list.append(l)

    for m, color in zip(output_ch_name, colors):
        # l = p2.line(inverter_data_norm_online.index, inverter_data_norm_online[m], alpha=0.5, line_color=color, legend_label=m)
        l = p1.line(inverter_warm.index, inverter_warm[m], alpha=0.5, line_color=color, legend_label=m)
        line_list.append(l)

    checkbox1 = CheckboxGroup(labels=input_ch_name + output_ch_name, active=[0, 1, 2, 3, 4, 5], width=100)

    callback1 = CustomJS(
        args=dict(l0=line_list[0], l1=line_list[1], l2=line_list[2], l3=line_list[3], l4=line_list[4], l5=line_list[5],
                  checkbox=checkbox1), code="""
    l0.visible = checkbox.active.includes(0);
    l1.visible = checkbox.active.includes(1);
    l2.visible = checkbox.active.includes(2);
    l3.visible = checkbox.active.includes(3);
    l4.visible = checkbox.active.includes(4);
    l5.visible = checkbox.active.includes(5);
    """)
    checkbox1.js_on_change('active', callback1)
    p1.legend.label_text_font_size = '10pt'
    p1.add_layout(p1.legend[0], 'right')

    path = path_dataset + '/' + f'inverter_{num_inverter}_warm_data_clean.html'
    # output_file(f"data_clean/inverter_{num_inverter}_warm_data_clean.html")
    output_file(path)
    save(column(row(p1, checkbox1), row(p2, checkbox2)))

def plot_for_paper(simu_sys, inverter_data):
    target_name = "Target"
    start_time_str = "2019-06-02"
    end_time_str = "2019-06-06"
    start_time = pd.to_datetime(start_time_str, utc=True)
    end_time = pd.to_datetime(end_time_str, utc=True)
    pred = simu_sys.pred_output_signals_mus[start_time_str:end_time_str]
    real = inverter_data[start_time_str:end_time_str]
    data_process = DataPreprocess()
    smooth_outlier_score_prob = data_process.smooth(simu_sys.outlier_info_list['outlier_prob'], 36)
    outlier_prob = simu_sys.outlier_info_list
    outlier_prob['smooth_outlier_prob'] = smooth_outlier_score_prob
    outlier_prob = outlier_prob[start_time_str:end_time_str]
    outlier_prob = outlier_prob['smooth_outlier_prob']

    pred = pred[target_name]
    real = real[target_name]
    plt.plot(pred.index, pred.values)
    plt.plot(real.index, real.values)
    plt.plot(outlier_prob.index, outlier_prob.values)
    tmp_anomaly_time = dict()
    tmp_anomaly_time['start']=[]
    tmp_anomaly_time['end'] = []
    for idx in range(len(simu_sys.anomaly_time['start'])):
        if simu_sys.anomaly_time['start'][idx]>start_time and simu_sys.anomaly_time['end'][idx]<end_time:
            #print(simu_sys.anomaly_time['start'][idx] - simu_sys.anomaly_time['end'][idx-1])
            if idx>0 and simu_sys.anomaly_time['start'][idx] - simu_sys.anomaly_time['end'][idx-1] < pd.Timedelta(minutes=20):
               print(simu_sys.anomaly_time['start'][idx],simu_sys.anomaly_time['end'][idx])
               tmp_anomaly_time['end'][-1]=simu_sys.anomaly_time['end'][idx]
            else:
                tmp_anomaly_time['start'].append(simu_sys.anomaly_time['start'][idx])
                tmp_anomaly_time['end'].append(simu_sys.anomaly_time['end'][idx])

    for idx in range(len(tmp_anomaly_time['start'])):
        plt.axvspan(tmp_anomaly_time['start'][idx], tmp_anomaly_time['end'][idx], alpha=0.5, color='red')

def db_gen(simu_sys, anomaly_clustering, warm_up_pred_mus, inverter_data, num_inverter, input_ch_name
         , output_ch_name, start, end, failure_time, path_dataset, smooth_par, method, synthetic_anomaly_type):
    target_name = "Target"
    epistemic_uncer = simu_sys.pred_output_signals_epistemic[target_name]
    # print(f'epistemic_uncer: {epistemic_uncer}')
    epistemic_uncer[epistemic_uncer > 10] = 10
    aleatoric_uncer = simu_sys.pred_output_signals_aleatoric[target_name]
    aleatoric_uncer[aleatoric_uncer > 10] = 10
    pred_mean = simu_sys.pred_output_signals_mus[target_name]
    data_process = DataPreprocess()
    outlier_info = simu_sys.outlier_info_list
    smooth_outlier_score_prob = data_process.smooth(outlier_info['outlier_prob'], smooth_par)
    outlier_info['smooth_outlier_prob'] = smooth_outlier_score_prob
    prediction_result_list = [pred_mean, epistemic_uncer, aleatoric_uncer, outlier_info['smooth_outlier_prob']]
    prediction_result = pd.concat(prediction_result_list,axis=1, join="inner")
    prediction_result.columns = [f'{target_name}_Pred_Mean', f'{target_name}_Epistemic', f'{target_name}_Aleatoric', 'Anomaly_Prob']
    #print(prediction_result.head(5))
    file_name = f'inverter_{num_inverter}_prediction_result'
    csv_path = path_dataset + '/' + f'{file_name}.csv'
    prediction_result.to_csv(csv_path)


    anomaly_result = dict()
    anomaly_result['anomaly_start'] = []
    anomaly_result['anomaly_end'] = []
    anomaly_result['cluster_type'] = []
    anomaly_result['is_real_anomaly'] = []
    
    for idx in range(len(simu_sys.novelty_time['start'])):
        start = simu_sys.novelty_time['start'][idx]
        end = simu_sys.novelty_time['end'][idx]
        cluster_type = anomaly_clustering.labels_[idx]
        anomaly_result['anomaly_start'].append(start)
        anomaly_result['anomaly_end'].append(end)
        anomaly_result['cluster_type'].append(cluster_type)
        anomaly_result['is_real_anomaly'].append(0)
    anomaly_result_df = pd.DataFrame.from_dict(anomaly_result)  
    file_name = f'inverter_{num_inverter}_anomaly_result'
    csv_path = path_dataset + '/' + f'{file_name}.csv'
    #print(anomaly_result_df.head(5))
    anomaly_result_df.to_csv(csv_path)

    file_name = f'inverter_{num_inverter}'
    db_path = path_dataset + '/' + f'{file_name}.db'  
    conn = sqlite3.connect(db_path)
    anomaly_result_df.to_sql('Anomaly_Result', conn, if_exists='replace', index=False)
    prediction_result.to_sql('Prediction_Result', conn, if_exists='replace', index=True)
    inverter_data.to_sql('Original_Data', conn, if_exists='replace', index=True)


def visualization(simu_sys, anomaly_clustering, warm_up_pred_mus, inverter_data, num_inverter, input_ch_name
                  , output_ch_name, start, end, failure_time, path_dataset, smooth_par, method, synthetic_anomaly_type):
    target_name = "Target"
    #plot_for_paper(simu_sys,inverter_data)
    epistemic_uncer = simu_sys.pred_output_signals_epistemic[target_name]
    # print(f'epistemic_uncer: {epistemic_uncer}')
    epistemic_uncer[epistemic_uncer > 10] = 10
    aleatoric_uncer = simu_sys.pred_output_signals_aleatoric[target_name]
    aleatoric_uncer[aleatoric_uncer > 10] = 10
    crp_scores = simu_sys.crp_score_list['crp_score']
    crp_scores[crp_scores > 10] = 10


    p = figure(title="Detection System", plot_width=1000, plot_height=300, x_axis_type='datetime')
    pred = p.line(simu_sys.pred_output_signals_mus.index, simu_sys.pred_output_signals_mus[target_name],
                  alpha=0.5, line_color="blue", legend_label=f'Pred {target_name}')
    pred_warm = p.line(warm_up_pred_mus.index, warm_up_pred_mus[f'{target_name}+1440Min'],
                  alpha=0.5, line_color="green", legend_label=f'Pred_warm {target_name}')
    real = p.line(inverter_data.index, inverter_data[target_name], alpha=0.5, line_color="darkorange", line_width=2,
                  legend_label=f'Real {target_name}')
    cal_score = p.line(simu_sys.calibration_date_list, simu_sys.calibration_score_list, alpha=0.5,
                       legend_label='calibration_score', line_color="green")
    data_process = DataPreprocess()
    smooth_outlier_score_prob = data_process.smooth(simu_sys.outlier_info_list['outlier_prob'], smooth_par)
    outlier_prob = p.line(simu_sys.outlier_info_list.index, smooth_outlier_score_prob, legend_label='outlier_prob',
                          line_color="red")

    crp_score = p.line(simu_sys.crp_score_list.index, crp_scores, legend_label='crp_score', alpha=0.5,
                       line_color="gray")
    aleatoric = p.line(simu_sys.pred_output_signals_aleatoric.index, aleatoric_uncer, alpha=0.5,
                       legend_label='aleatoric',
                       line_color="darkgray")
    epistemic = p.line(simu_sys.pred_output_signals_aleatoric.index, epistemic_uncer, alpha=0.5,
                       legend_label='epistemic',
                       line_color="lightgray")
    if len(failure_time) != 0:
        vline = Span(location=failure_time[0], dimension='height', line_color='green', line_width=3)
    else:
        vline = Span(location=0, dimension='height', line_color='green', line_width=3)
    p.add_layout(vline)

    # vline_warm_online = Span(location=online_start, dimension='height', line_color='red', line_width=3)
    # p.add_layout(vline_warm_online)
    #anomaly_clustering = simu_sys.get_anomaly_cluster()
    cmap = ['blue', 'green', 'cyan', 'magenta', 'yellow', 'darkblue', 'darkgreen', 'darkcyan', 'darkmagenta', 'darkyellow']
    for idx in range(len(simu_sys.anomaly_time['start'])):
        start_float = (simu_sys.anomaly_time['start'][idx]).timestamp() * 1000
        end_float = (simu_sys.anomaly_time['end'][idx]).timestamp() * 1000
        cluster_type = anomaly_clustering.labels_[idx]
        #if idx==31:
        #    print(simu_sys.anomaly_time['start'][idx], simu_sys.anomaly_time['end'][idx], cluster_type, cmap[cluster_type])
        if cluster_type == -1:
            v_box = BoxAnnotation(left=start_float, right=end_float, bottom=-0.1, top=1.1, fill_alpha=0.2,
                                  fill_color='red')
        else:
            v_box = BoxAnnotation(left=start_float, right=end_float, bottom=-0.1, top=1.1, fill_alpha=0.2,
                                  fill_color=cmap[cluster_type])
        p.add_layout(v_box)

    checkbox = CheckboxGroup(
        labels=[f'Pred {output_ch_name[0]}', f'Real {output_ch_name[0]}', "outlier_prob", "calibration_score", "crp_score", "aleatoric",
                "epistemic", "real_failure", f'Pred_warmup {output_ch_name[0]}'],
        active=[0, 1, 2, 3, 4, 5, 6, 7, 8], width=100)

    callback = CustomJS(
        args=dict(l0=pred, l1=real, l2=outlier_prob, l3=cal_score, l4=crp_score, l5=aleatoric, l6=epistemic, l7=vline,
                  l8=pred_warm,
                  checkbox=checkbox), code="""
                l0.visible = checkbox.active.includes(0);
                l1.visible = checkbox.active.includes(1);                
                l2.visible = checkbox.active.includes(2);
                l3.visible = checkbox.active.includes(3);
                l4.visible = checkbox.active.includes(4);
                l5.visible = checkbox.active.includes(5);
                l6.visible = checkbox.active.includes(6);
                l7.visible = checkbox.active.includes(7);
                l8.visible = checkbox.active.includes(8);
                """)

    checkbox.js_on_change('active', callback)
    p.legend.label_text_font_size = '10pt'
    p.add_layout(p.legend[0], 'right')
    p.y_range = Range1d(-0.1, 1.1)

    p1 = figure(title="Alarm System", plot_width=1000, plot_height=300, x_range=p.x_range, y_range=p.y_range,
                x_axis_type='datetime')
    pred = p1.line(simu_sys.pred_output_signals_mus.index, simu_sys.pred_output_signals_mus[target_name],
                   alpha=0.5, line_color="blue", legend_label=f'Pred {target_name}')
    pred_warm = p1.line(warm_up_pred_mus.index,
                       warm_up_pred_mus[f'{target_name}+1440Min'],
                       alpha=0.5, line_color="green", legend_label=f'Pred_warm {target_name}')
    max_legend_len = len(f'Pred_warm {target_name}')
    p1.legend.label_text_font_size = '10pt'
    p1.y_range = Range1d(-0.1, 1.1)
    aleatoric = p1.line(simu_sys.pred_output_signals_aleatoric.index, aleatoric_uncer, alpha=0.5,
                        legend_label='aleatoric',
                        line_color="darkgray")
    epistemic = p1.line(simu_sys.pred_output_signals_aleatoric.index, epistemic_uncer, alpha=0.5,
                        legend_label='epistemic',
                        line_color="lightgray")
    checkbox1 = CheckboxGroup(labels=[f"Pred {target_name}", "aleatoric", "epistemic", f'Pred_warmup {target_name}'],
                              active=[0, 1, 2, 3, 4, 5, 6], width=100)
    callback1 = CustomJS(args=dict(l0=pred, l1=aleatoric, l2=epistemic, l3=pred_warm, checkbox=checkbox1), code="""
            l0.visible = checkbox.active.includes(0);
            l1.visible = checkbox.active.includes(1);
            l2.visible = checkbox.active.includes(2);
            l3.visible = checkbox.active.includes(3);
            """)
    checkbox1.js_on_change('active', callback1)
    p1.legend.label_text_font_size = '10pt'
    p1.add_layout(p1.legend[0], 'right')

    line_list = []
    p2 = figure(title="Input Sig", plot_width=1000, plot_height=300, x_range=p.x_range, y_range=p.y_range,
                x_axis_type='datetime')
    colors = itertools.cycle(palette)

    for m, color in zip(input_ch_name, colors):
        # l = p2.line(inverter_data_norm_online.index, inverter_data_norm_online[m], alpha=0.5, line_color=color, legend_label=m)
        l = p2.line(inverter_data.index, inverter_data[m], alpha=0.5, line_color=color, legend_label=m + ' ' * (max_legend_len-len(m)+4))
        line_list.append(l)
    outlier_scores_input = p2.line(simu_sys.outlier_score_input.index, simu_sys.outlier_score_input['outlier_score_input'],
                                   alpha=0.5,
                                   legend_label='outlier_score_input',
                                   line_color="blue"
                                   )
    # line_list.append(outlier_score_input)
    input_ch_name.append('outlier_score_input')
    #print(f'input_ch_name_labels: {input_ch_name}')
    # checkbox2 = CheckboxGroup(labels=input_ch_name, active=[0, 1, 2, 3, 4, 5], width=100)
    checkbox2 = CheckboxGroup(labels=input_ch_name, active=[0, 1, 2, 3, 4, 5], width=100)

    callback2 = CustomJS(
        args=dict(l0=line_list[0], l1=line_list[1], l2=line_list[2], l3=line_list[3], l4=line_list[4], l5=outlier_scores_input,
                  checkbox=checkbox2),
        code="""
        l0.visible = checkbox.active.includes(0);
        l1.visible = checkbox.active.includes(1);
        l2.visible = checkbox.active.includes(2);
        l3.visible = checkbox.active.includes(3);
        l4.visible = checkbox.active.includes(4);
        l5.visible = checkbox.active.includes(5);
        """)
    checkbox2.js_on_change('active', callback2)
    p2.legend.label_text_font_size = '10pt'
    p2.add_layout(p2.legend[0], 'right', )
    file_name = f'inverter_{num_inverter}_{start}_{end}'
    if num_inverter == "artificial_dataset":
        file_name = f'inverter_{num_inverter}_synthetic_anomaly_type_{synthetic_anomaly_type}_{start}_{end}'
    if method == 'VEHDNN':
        path = path_dataset + '/' + f'{file_name}_VEHDNN.html'
    elif method == 'VEHLSTM':
        path = path_dataset + '/' + f'{file_name}_VEHLSTM.html'
    else:
        path = path_dataset + '/' + f'{file_name}_MCDHDNN.html'
    # output_file(f"data_clean/inverter_{num_inverter}_warm_data_clean.html")
    output_file(path)
    # save(column(row(p1, checkbox1), row(p2, checkbox2)))
    # output_file(f"./output/inverter_{num_inverter}_{start}_{end}.html")
    save(column(row(p1, checkbox1), row(p, checkbox), row(p2, checkbox2)))

def shap_explain(simu_sys, inverter_data_norm_online, input_ch_name, output_ch_name, time_start, time_end, time_middle, smooth_parm, save_path):
    data_process = DataPreprocess()
    smooth_outlier_score_prob = data_process.smooth(simu_sys.outlier_info_list['outlier_prob'], 18)

    X = inverter_data_norm_online[simu_sys.outlier_info_list.index[0]:simu_sys.outlier_info_list.index[-1]][
        input_ch_name + output_ch_name]
    y = smooth_outlier_score_prob
    # train an XGBoost models
    model = xgboost.XGBRegressor().fit(X, y)
    probas_xgb = pd.Series(model.predict(X), index=X.index)
    explainer = shap.Explainer(model)
    shap_values = explainer(X)
    shap_df = pd.DataFrame([list(shap_values[n].values) for n in range(X.shape[0])], columns=X.columns,
                           index=X.index)

    shap.initjs()
    shap.save_html(f"{save_path}/shap_time_interval.html",
                   shap.plots.force(explainer.expected_value, shap_df[time_start:time_end].values
                                    , X[time_start:time_end].values, feature_names=X.columns))
    idxs = shap_df.index.get_loc(time_middle)
    print(idxs)
    shap.save_html(f"{save_path}/shap_point.html",
                   shap.force_plot(explainer.expected_value, shap_values.values[idxs, :], X.iloc[idxs, :]))

def anomaly_clustering_plot(anomaly_plot, anomaly_list, cluster2anomaly=None):
    cluster_num = np.unique(anomaly_plot['cluster_result']).size
    anomaly_ids = np.unique(anomaly_plot['cluster_result'])

    ncol = 2
    nrow = math.ceil(cluster_num / ncol) #int(cluster_num+0.5 / ncol)
    print(anomaly_ids, nrow)
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol)
    cmap = ['blue', 'green', 'cyan', 'magenta', 'yellow', 'darkblue', 'darkgreen', 'darkcyan', 'darkmagenta', 'darkyellow']
    fig.suptitle(("Anomalies Cluster"))
    for yi in range(-1, len(anomaly_ids)):
        novelties = anomaly_plot[anomaly_plot['cluster_result'] == yi]
        for idx, novelty in novelties.iterrows():
            if yi == -1:
                axs.reshape(-1)[yi].plot(anomaly_list[idx][:, 0], "-", alpha=.2, color='red')
                axs.reshape(-1)[yi].plot(anomaly_list[idx][:, 1], "-", alpha=.2, color='darkred')
                axs.reshape(-1)[yi].patch.set_facecolor('red')
                axs.reshape(-1)[yi].patch.set_alpha(alpha=.1)
                #axs.reshape(-1)[yi].set_ylim(-1, 1)
                pass
            else:
                tmp_yi = yi
                if cluster2anomaly is not None:
                    tmp_yi = cluster2anomaly[yi]-1
                axs.reshape(-1)[tmp_yi].plot(anomaly_list[idx][:, 0], "-", alpha=.1, color='blue')
                axs.reshape(-1)[tmp_yi].plot(anomaly_list[idx][:, 1], "-", alpha=.1, color='darkorange')
                axs.reshape(-1)[tmp_yi].patch.set_facecolor(cmap[yi])
                axs.reshape(-1)[tmp_yi].patch.set_alpha(alpha=.1)
                if cluster2anomaly is not None:
                    axs.reshape(-1)[tmp_yi].set_title(f"A_{tmp_yi}")
    for i in range(cluster_num):
        if i == 1:
            axs.reshape(-1)[i].plot([], [], "-", color='blue', label="pred")
            axs.reshape(-1)[i].plot([], [], "-", color='darkorange', label="real")
            axs.reshape(-1)[i].legend(loc="upper right")
    plt.show()

def anomaly_clustering_plot_single(anomaly_plot, anomaly_list, plot_num=10, type_id=0):
    anomaly_ids = np.unique(anomaly_plot['cluster_result'])
    ncol = 5
    nrow = math.ceil(plot_num / ncol)
    print(anomaly_ids, nrow)
    novelties = anomaly_plot[anomaly_plot['cluster_result'] == type_id]
    idx_list = []
    cmap = ['blue', 'green', 'cyan', 'magenta', 'yellow', 'darkblue', 'darkgreen', 'darkcyan', 'darkmagenta',
            'darkyellow']
    for idx, novelty in novelties.iterrows():
        idx_list.append(idx)
    selected_idx = np.random.choice(idx_list, plot_num)
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol, figsize=(3*nrow, 3*ncol))
    fig, axs = plt.subplots(nrows=nrow, ncols=ncol)
    for i in range(plot_num):
        axs.reshape(-1)[i].plot(anomaly_list[selected_idx[i]][:, 0], "-", alpha=1.0, color='blue')
        axs.reshape(-1)[i].plot(anomaly_list[selected_idx[i]][:, 1], "-", alpha=1.0, color='darkorange')
        axs.reshape(-1)[i].patch.set_facecolor(cmap[type_id])
        axs.reshape(-1)[i].patch.set_alpha(alpha=.1)
        axs.reshape(-1)[i].set_ylim(0,1)
        axs.reshape(-1)[i].set_xlim(0,50)
        if i % ncol!=0:
            axs.reshape(-1)[i].get_yaxis().set_ticks([])
    plt.show()

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    """
    given a sklearn confusion matrix (cm), make a nice plot

    Arguments
    ---------
    cm:           confusion matrix from sklearn.metrics.confusion_matrix

    target_names: given classification classes such as [0, 1, 2]
                  the class names, for example: ['high', 'medium', 'low']

    title:        the text to display at the top of the matrix

    cmap:         the gradient of the values displayed from matplotlib.pyplot.cm
                  see http://matplotlib.org/examples/color/colormaps_reference.html
                  plt.get_cmap('jet') or plt.cm.Blues

    normalize:    If False, plot the raw numbers
                  If True, plot the proportions

    Usage
    -----
    plot_confusion_matrix(cm           = cm,                  # confusion matrix created by
                                                              # sklearn.metrics.confusion_matrix
                          normalize    = True,                # show proportions
                          target_names = y_labels_vals,       # list of names of the classes
                          title        = best_estimator_name) # title of graph

    Citiation
    ---------
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html

    """
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.tight_layout()
    plt.show()