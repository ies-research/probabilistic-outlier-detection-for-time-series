import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from numpy.random import randn, uniform
import copy
import math
class DataPreprocess:
    def __init__(self):
        pass

    # normalize the raw data. Return normalized data frame and scaler
    def normalization(self, df_origin_warm, df_origin_online, scaler=None):
        """
        df_origin_warm: pd.dataframe, raw dataframe for warmup with the resolution of 10 min
        df_origin_online: pd.dataframe, raw dataframe for online detection with the resolution of 10 min
        scaler: object, scaler for warmup data
        return:
        df_norm_warm: pd.dataframe, the scaled dataframe for warmup training.
        df_norm_online: pd.dataframe, the scaled dataframe for online detection.
        scaler: object, scaler for warmup data
        """
        if scaler is None:
            scaler = MinMaxScaler()

        data_norm_warm = scaler.fit_transform(df_origin_warm)
        df_norm_warm = pd.DataFrame(data=data_norm_warm, index=df_origin_warm.index, columns=df_origin_warm.columns)

        data_norm_online = scaler.transform(df_origin_online)
        df_norm_online = pd.DataFrame(data=data_norm_online, index=df_origin_online.index,
                                      columns=df_origin_online.columns)

        return df_norm_warm, df_norm_online, scaler

    # Check if the channel is a numeric channel
    def is_enum_channel(self, one_channel_dframe):
        """
        one_channel_dframe: pd.dataframe, one column of a raw dataframe
        return:
        is_enum_channel: bool, if it is a category channel.
        """
        if one_channel_dframe.dtype.name == 'category' or one_channel_dframe.dtype.name == 'object':
            if isinstance(one_channel_dframe.unique()[0], str):
                return True
        return False

    # smooth a series
    def smooth(self, y, box_pts=10):
        """
        y: numpy.array, a 1 dim raw series data
        box_pts: int, smooth window
        return:
        is_enum_channel: bool, if it is a category channel.
        """
        box = np.ones(box_pts) / box_pts
        y_smooth = np.convolve(y, box, mode='same')
        return y_smooth
    
    def artificial_ts(self, time, periods, resolution, num_features, synthetic_anomaly_confs, seed=12345):
        np.random.seed(seed)
        N = len(periods)

        param_ts = {'amplitude': [],
                    'phase': []}
        X = []

        for j in range(num_features):
            amlitude = (randn(N) * [1, 10]) + [5, 20]
            phase = randn(N)
            #phase = uniform(0,2*math.pi, N)
            # x_j generation

            x_uni = [amlitude[i] * np.sin(2 * np.pi * time / periods[i] + phase[i]) for i in range(N)]
            X.append(np.mean(x_uni, axis=0))

            param_ts['amplitude'].append(amlitude)
            param_ts['phase'].append(phase)

        y = np.mean(X, axis=0) ** 3

        X = np.array(X)[:, ::resolution].T
        y = np.array(y).reshape(-1, 1)[::resolution, :]

        X_noise = randn(X.shape[0], X.shape[1])
        X_noisy = X + X_noise
        y_noise = randn(y.shape[0], y.shape[1]) *5
        
        y_noisy = y + y_noise

        time_series_noisy = np.concatenate((X_noisy, y_noisy), axis=1)
        time_series_noisy = pd.DataFrame(time_series_noisy)
        try:
            Timestamp = pd.date_range(start='2018-01-01', end='2020-01-01', freq='10min', inclusive='left')
        except:
            Timestamp = pd.date_range(start='2018-01-01', end='2020-01-01', freq='10min')
            Timestamp = Timestamp[:-1]

        time_series_noisy.insert(0, "Timestamp", Timestamp, False)
        time_series_noisy.columns = ['Timestamp', "Feature_1", "Feature_2", "Feature_3", "Feature_4",
                                       "Feature_5", "Target"]
        time_series_noisy["Timestamp"] = pd.to_datetime(time_series_noisy["Timestamp"])
        time_series_noisy = time_series_noisy.set_index("Timestamp")
        time_series_noisy = time_series_noisy.sort_index()
        time_series_noisy = self.add_synthetic_anomaly(time_series_noisy, synthetic_anomaly_confs)
        scaler = MinMaxScaler()
        train_set_scaled = scaler.fit_transform(time_series_noisy.values[:, :])
        train_set_scaled_df = pd.DataFrame(train_set_scaled
                                           , columns=time_series_noisy.keys()
                                           , index=time_series_noisy.index)
        return train_set_scaled_df,scaler
    
    def load_artificial_data(self,data_conf, synthetic_anomaly_confs):
        YEARS = 2
        DAYS = 365 * YEARS  # 730 days
        UNIT = 1  # 1 unit = 1 mins
        PERIOD_D = 24 * 60  # one day period (number of units in one day)
        PERIOD_Y = 24 * 60 * 365  # one year period (number of units in one year)
        RESOLUTION = 10 * UNIT

        time = np.arange(0, DAYS * PERIOD_D, UNIT)
        inverter_df, scaler = self.artificial_ts(time, [PERIOD_D, PERIOD_Y], RESOLUTION, 5, synthetic_anomaly_confs)
        return inverter_df

    def load_inverter_norm_data(self, path_dataset):
        inverter_data_norm = pd.read_csv(f'{path_dataset}')
        inverter_data_norm["Timestamp"] = pd.to_datetime(inverter_data_norm["Timestamp"])
        inverter_data_norm = inverter_data_norm.set_index("Timestamp")
        inverter_data_norm = inverter_data_norm.sort_index()
        return inverter_data_norm

    # Load inverter data, pad missing values with 0
    def load_inverter_data(self, path_dataset):
        """
        path_dataset: str, path of dataset file
        path_failure_info: str, path of file with failure timestamp
        num_inverter: int, inverter id.
        return:
        inverter_df_numeric: pd.dataframe, numeric channel data
        inverter_df_category: pd.dataframe, category channel data
        """
        inverter_df = pd.read_csv(path_dataset)
        inverter_df["Timestamp"] = pd.to_datetime(inverter_df["Timestamp"])
        inverter_df = inverter_df.set_index("Timestamp")
        inverter_df = inverter_df.sort_index()
        category_channel_list = []
        numeric_channel_list = []
        for ch_name in inverter_df.keys():
            if self.is_enum_channel(inverter_df[ch_name]):
                inverter_df[ch_name] = inverter_df[ch_name].astype('category')
                category_channel_list.append(ch_name)
                inverter_df[ch_name + '_v'] = inverter_df[ch_name].cat.codes
                # inverter_df.drop(ch_name, axis='columns', inplace=True)
                numeric_channel_list.append(ch_name + '_v')
            else:
                numeric_channel_list.append(ch_name)

        inverter_df_numeric = inverter_df[numeric_channel_list]
        inverter_df_category = inverter_df[category_channel_list]

        inverter_df_numeric = inverter_df_numeric.fillna(0)
        full_index = pd.date_range(inverter_df_numeric.index[0], inverter_df_numeric.index[-1],
                                   freq=pd.DateOffset(minutes=10))
        inverter_df_numeric = inverter_df_numeric.reindex(full_index, fill_value=0)

        return inverter_df_numeric, inverter_df_category

    # Create time sequence, forecast next N hour's data
    def build_time_sequence(self, raw_data, features_cols, targets_cols, nr_in_steps, nr_out_steps, interval,
                            autoregression, resolution=10):
        """
        raw_data: pd.dataframe, the filled, scaled dataframe with the resolution of 10 min
        features_cols: list, containing the name of the selected column as the input features
        targets_cols: list, containing the name of the selected column as the output targets
        nr_in_steps: int, the number of the input historical observation
        nr_out_steps: int, the number of the output target observation
        interval: int, the interval between two observations, the unit of the interval is the same as the resolution
        autoregression: boolean, using lagged observations of the target if True.
        resolution: int, the resolution of the given dataframe, default as 10 min
        """
        features_sequence = []
        targets_sequence = []

        if autoregression:
            features_cols = features_cols + targets_cols

        for col in features_cols:
            time_series = pd.concat([raw_data[col].shift(step * interval) for step in range(nr_in_steps)], axis=1)
            time_series.columns = [f'{col}-{i * resolution * interval}Min' for i in range(nr_in_steps)]
            features_sequence.append(time_series)
        features_sequence = pd.concat(features_sequence, axis=1).dropna()

        for col in targets_cols:
            time_series = pd.concat([raw_data[col].shift(-step * interval) for step in range(nr_out_steps + 1)], axis=1)
            time_series.columns = [f'{col}+{i * resolution * interval}Min' for i in range(nr_out_steps + 1)]
            targets_sequence.append(time_series)
        targets_sequence = pd.concat(targets_sequence, axis=1).dropna()

        features_sequence = features_sequence.reindex(targets_sequence.index).dropna()
        targets_sequence = targets_sequence.reindex(features_sequence.index).dropna()

        # Adding temporal feature: the hour of the prediction time
        # Scaling by sin/cosine-coding
        hour = features_sequence.index.hour.values
        hour_cos = 0.5 * np.cos(np.pi * hour / 12) + 0.5
        hour_sin = 0.5 * np.sin(np.pi * hour / 12) + 0.5
        features_sequence['Hour_cos'] = hour_cos
        features_sequence['Hour_sin'] = hour_sin

        return features_sequence, targets_sequence


    def synthetic_anomaly(self, type_idx, time_series, alpha, beta, b=0, increasing=True):
        # try:
        #     length_ts = time_series.shape[0]
        #     channel_num = time_series.shape[1]
        # except:
        #     time_series = time_series.reshape(-1, 1)
        #     length_ts = time_series.shape[0]
        #     channel_num = time_series.shape[1]
        #
        # samples_gaussian = np.random.randn(length_ts, channel_num)

        length_ts = time_series.shape[0]
        channel_num = time_series.shape[1]
        samples_gaussian = np.random.randn(length_ts, channel_num)

        if type_idx == 1:

            return time_series[0, :] * alpha + samples_gaussian * beta

        elif type_idx == 2:

            return alpha * time_series + b + samples_gaussian * beta

        elif type_idx == 3:

            if increasing:
                tau = np.arange(1, length_ts + 1)
            else:
                tau = np.arange(length_ts, 0, -1)

            return alpha * time_series[0, :] + beta * (np.sqrt(tau).T * samples_gaussian.T).T

        elif type_idx == 4:
            tau = np.arange(1, length_ts + 1).repeat(channel_num).reshape(length_ts, channel_num)
            return alpha * (np.sqrt(tau) + time_series[0, :]) + b + beta * (np.sqrt(tau).T * samples_gaussian.T).T


    def add_synthetic_anomaly(self, time_series, synthetic_anomaly_confs, seed=45678):
        np.random.seed(seed)
        all_input_channels = []

        for sa_conf in synthetic_anomaly_confs:
            if sa_conf["type"]=="real":
                continue
            # only for sgao###
            start = time_series.loc[[sa_conf['start']]].index.tolist()[0]
            end = time_series.loc[[sa_conf['end']]].index.tolist()[0]
            org_series = time_series[start:end].copy()
            replace_series = self.synthetic_anomaly(sa_conf['anomaly_type'], org_series[sa_conf['in_channel']].values,
                                                    increasing=sa_conf['increasing'], alpha=sa_conf['alpha'],
                                                    beta=sa_conf['beta'], b=sa_conf['b']) + org_series[
                                                    sa_conf['in_channel']]
            #print(f'replace_series: {replace_series}')
            time_series[start:end][sa_conf['in_channel']] = replace_series
            #time_series.loc[start:end, (sa_conf['in_channel'])] = replace_series


            time_series[start:end][sa_conf['out_channel'][-1]] = time_series[start:end][sa_conf['in_channel']].mean(axis=1).values.reshape(-1)**3
            #time_series.loc[start:end, (sa_conf['out_channel'][-1])] = time_series[start:end][sa_conf['in_channel']].mean(
            #    axis=1).values.reshape(-1)**3

        return time_series
    '''

    def add_synthetic_anomaly(self, time_series, synthetic_anomaly_confs, scaler, seed=45678):
        np.random.seed(seed)
        all_input_channels = []
        for sa_conf in synthetic_anomaly_confs:
            if sa_conf["type"] == "real":
                continue
            # only for sgao###
            start = time_series.loc[[sa_conf['start']]].index.tolist()[0]
            end = time_series.loc[[sa_conf['end']]].index.tolist()[0]
            org_series = time_series[start:end].copy()
            import matplotlib.pyplot as plt
            plt.figure(figsize=(16, 6))
            org_series_data = scaler.inverse_transform(org_series[sa_conf['in_channel']].values[:, :])
            org_series_data_df = pd.DataFrame(org_series_data, columns=time_series.keys())
            plt.plot(org_series_data_df.index, org_series_data_df.values, alpha=0.5)
            plt.show()
            replace_series = self.synthetic_anomaly(sa_conf['anomaly_type'], org_series[sa_conf['in_channel']].values,
                                               increasing=sa_conf['increasing'], alpha=sa_conf['alpha'],
                                               beta=sa_conf['beta'], b=sa_conf['b']) + org_series[
                                 sa_conf['in_channel']]
            # print(f'replace_series: {replace_series}')
            time_series[start:end][sa_conf['in_channel']] = replace_series
            time_series.loc[start:end, (sa_conf['in_channel'])] = replace_series


            org_scale_data = scaler.inverse_transform(time_series[start:end].values[:, :])
            org_scale_data_df = pd.DataFrame(org_scale_data, columns=time_series.keys())
            org_scale_output_data = org_scale_data_df[sa_conf['in_channel']].values.mean(axis=1).reshape(-1) ** 3
            org_scale_data_df[sa_conf['out_channel'][-1]] = org_scale_output_data
            re_scale_data = scaler.transform(org_scale_data_df.values)
            re_scale_data_df = pd.DataFrame(re_scale_data, columns=time_series.keys())
            time_series[start:end][sa_conf['out_channel'][-1]] = re_scale_data_df[sa_conf['out_channel'][-1]].values
            time_series.loc[start:end, (sa_conf['out_channel'][-1])] = re_scale_data_df[sa_conf['out_channel'][-1]].values
        return time_series
    '''

    def interpolation(self, original_df, input_ch_names: list, output_ch_name: str, days=7, clean_input=True):
        """
        The function is used to clean the continuous zero values in the output channel of the training dataset.
        The length of the continuous zero section must be within the value of the input variable days, otherwise
        the interpulation function fails.

        original_df: pd.DataFrame, the original inverter dataset
        input_ch_names: list, the list containing input channel names
        output_ch_name: string, the name of the single target column
        days: int, averaging the data of former/latter days within the same period for interpolation
        clean_input: bool, clean the input channels with the same data segment of interpulating the output channel if True.

        return:
        clean_df: pd.Dataframe, the cleaned original data frame
        itp_index_all_sects: list, the start and end points of all interpolation segments for all zero sections, None means the
                             corresponding segment doesn't exist.
        sect_starts: list, contains the start points of the zero sections
        sect_ends: list, contains the end points of the zero sections

        """
        clean_df = copy.copy(original_df)

        target_series = clean_df[output_ch_name]

        # locate the indices where zero is.
        zero_indices = original_df[output_ch_name].loc[(original_df[output_ch_name] == 0.0).values].index

        ######################################## INQUIRY CONTINUOUS ZEROS SECTIONS #####################################
        try:
            # find the start index and the end index of the zero sections that contains continuous zero values
            sect_starts = []
            sect_ends = []
            sect_starts.append(zero_indices[0])

            for zero_idx_head, zero_idx_bottom in zip(zero_indices[:-1], zero_indices[1:]):
                if zero_idx_bottom - zero_idx_head > pd.Timedelta(10, unit='m'):
                    sect_ends.append(zero_idx_head)
                    sect_starts.append(zero_idx_bottom)

            sect_ends.append(zero_indices[-1])

        except:
            print('There is no outliers (zeros) in the target column.')
            itp_index_all_sects = []
        else:
            ######################################## INTERPOLATE THE TARGET SERIES #####################################
            itp_index_all_sects = []  # the indices of the selected interpolation segments for all zero sections

            for i in range(len(sect_starts)):
                duration = (sect_ends[i] - sect_starts[i]) // pd.Timedelta(10, unit='m') + 1
                interpolations = []

                itp_index_single_sect = []  # the indices of the selected interpolation segments for one zero section
                try:
                    for day in range(1, days + 1):
                        interpolation_start = sect_starts[i] - pd.Timedelta(day, unit='d')
                        interpolation_end = sect_ends[i] - pd.Timedelta(day, unit='d')

                        # The section for interpolation has sufficient measurements and doesn't have any zero.
                        if target_series.loc[interpolation_start:interpolation_end].shape[0] == duration and np.sum(
                                (target_series.loc[interpolation_start:interpolation_end] == 0).values) == 0:
                            interpolations.append(
                                target_series.loc[interpolation_start:interpolation_end].values.flatten())
                            itp_index_single_sect.append(interpolation_start)
                            itp_index_single_sect.append(interpolation_end)
                    if not interpolations:
                        raise Exception('interpolations is empty')
                except:
                    print(f'Error exists when interpolating {output_ch_name} using the data of the previous week')
                    try:
                        for day in range(1, 8):
                            interpolation_start = sect_starts[i] + pd.Timedelta(day, unit='d')
                            interpolation_end = sect_ends[i] + pd.Timedelta(day, unit='d')

                            # The section for interpolation has sufficient measurements and doesn't have any zero.
                            if target_series.loc[interpolation_start:interpolation_end].shape[0] == duration and np.sum(
                                    (target_series.loc[interpolation_start:interpolation_end] == 0).values) == 0:
                                interpolations.append(
                                    target_series.loc[interpolation_start:interpolation_end].values.flatten())
                                itp_index_single_sect.append(interpolation_start)
                                itp_index_single_sect.append(interpolation_end)
                        if not interpolations:
                            raise Exception('interpolations is empty')
                    except:
                        print(f'Error exists when interpolating {output_ch_name} using the data of the subsequent week')
                    else:
                        interpolation_mean = np.mean(interpolations, axis=0).reshape(-1, 1)
                else:
                    interpolation_mean = np.mean(interpolations, axis=0).reshape(-1, 1)

                try:
                    target_series.loc[sect_starts[i]:sect_ends[i]] = interpolation_mean
                except:
                    print(
                        f'Manually interpolating {output_ch_name} for the section between {sect_starts[i]} and {sect_ends[i]}')
                    itp_index_single_sect = None

                itp_index_all_sects.append(itp_index_single_sect)

        clean_df[output_ch_name] = target_series

        if clean_input:
            # CLEAN the input channels
            for input_ch_n in input_ch_names:
                target_series = clean_df[input_ch_n]

                for itp_idx in range(len(itp_index_all_sects)):
                    if itp_index_all_sects[itp_idx] is None:
                        # zero section exists but interpulation must be manually constructed
                        raise Exception(
                            f'Manually interpolating {output_ch_name} for the section between {sect_starts[itp_idx]} and {sect_ends[itp_idx]}')
                    else:
                        target_series.loc[sect_starts[itp_idx]:sect_ends[itp_idx]] = np.mean(
                            [target_series.loc[i:j].values for i, j in zip(itp_index_all_sects[itp_idx][0:-1:2],
                                                                           itp_index_all_sects[itp_idx][1:-1:2])],
                            axis=0)
                clean_df[input_ch_n] = target_series

        return clean_df, itp_index_all_sects, sect_starts, sect_ends
