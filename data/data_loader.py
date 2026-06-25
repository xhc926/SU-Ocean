import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.timefeatures import time_features
from utils.tools import StandardScaler


def _read_table(path):
    if path.endswith('.csv'):
        return pd.read_csv(path)
    return pd.read_pickle(path)


def _detect_time_col(df):
    """Return the time column name present in the DataFrame, preferring 'date' over 'time'.
    
    Also handles the case where time information is stored as the DataFrame index
    (e.g. DatetimeIndex) rather than as a named column.
    
    Returns:
        str  -- name of the time column in df.columns (always 'date' after normalization)
        None -- time is in the index (caller should reset_index)
    """
    cols = list(df.columns)
    for candidate in ['date', 'time', 'datetime', 'Date', 'Time']:
        if candidate in cols:
            return candidate

    # Check if time is stored as the index
    idx_name = df.index.name or ''
    if idx_name.lower() in ['date', 'time', 'datetime']:
        return None  # signal: time is in the index with a recognized name

    if isinstance(df.index, pd.DatetimeIndex):
        return None  # signal: time is in the index (DatetimeIndex)

    raise KeyError(
        "DataFrame must have a 'date'/'time' column or a DatetimeIndex, "
        "got columns={} index_name={}".format(cols, idx_name)
    )


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        if size is None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        _is_multi_factor = (self.root_path in ['./data/ALL/', '/root/autodl-tmp/data/upsampled/1-4/area1',
                                                 '/root/autodl-tmp/data/upsampled/1-4/area2',
                                                 '/root/autodl-tmp/data/upsampled/1-12/area3',
                                                 '/root/autodl-tmp/data/bohai']) and \
            self.data_path in ['ALL1.pkl', 'ALL2.pkl', 'ALL3.pkl', 'ALL4.pkl', 'ALL5.pkl', 'ALL6.pkl']

        if _is_multi_factor:
            self.scaler = []
        else:
            self.scaler = StandardScaler()

        if _is_multi_factor:
            if self.data_path == 'ALL1.pkl':
                base = '/root/autodl-tmp/data/upsampled/1-4/area1'
                pathlist = [[base, 'swh.pkl'], [base, 'u10.pkl'], [base, 'v10.pkl']]
            elif self.data_path == 'ALL2.pkl':
                base = '/root/autodl-tmp/data/upsampled/1-4/area2'
                pathlist = [[base, 'swh.pkl'], [base, 'u10.pkl'], [base, 'v10.pkl']]
            elif self.data_path == 'ALL3.pkl':
                # Three factors for iTransformerUHSM: sal, ssh, sst (channel order 0,1,2).
                base = self.root_path.rstrip(os.sep)
                pathlist = [[base, 'sal.pkl'], [base, 'ssh.pkl'], [base, 'sst.pkl']]
            elif self.data_path == 'ALL4.pkl':
                # Four factors for iTransformerUHSM4: sal, ssh, uo, vo (channel order 0..3).
                base = self.root_path.rstrip(os.sep)
                pathlist = [[base, 'sal.pkl'], [base, 'ssh.pkl'], [base, 'uo.pkl'], [base, 'vo.pkl']]
            elif self.data_path == 'ALL5.pkl':
                base = '/root/autodl-tmp/data/bohai'
                pathlist = [[base, 'u10.pkl'], [base, 'v10.pkl'], [base, 'uo.pkl'], [base, 'vo.pkl'], 
                            [base, 'VHM0.pkl'], [base, 'VMDR_cos.pkl'], [base, 'VMDR_sin.pkl'], [base, 'VTM02.pkl']]
            else:
                pathlist = [['./data/OISST/', 'OISST5.pkl'], ['./data/OISSS/', 'OISSS3.pkl']]

            raw_tables = []
            for root, name in pathlist:
                filepath = os.path.join(root, name)
                df_raw = _read_table(filepath)
                raw_tables.append((filepath, df_raw))
            min_rows = min(len(df) for _, df in raw_tables)

            data_x, data_y = [], []
            for filepath, df_raw in raw_tables:
                if len(df_raw) > min_rows:
                    df_raw = df_raw.iloc[:min_rows].copy()

                # Normalize time column: rename 'time' → 'date' for internal consistency
                time_col = _detect_time_col(df_raw)
                if time_col is None:
                    # Time is in the index — promote to a 'date' column
                    df_raw = df_raw.reset_index()
                    if df_raw.columns[0] != 'date':
                        df_raw = df_raw.rename(columns={df_raw.columns[0]: 'date'})
                elif time_col != 'date':
                    df_raw = df_raw.rename(columns={time_col: 'date'})

                if self.features == 'M':
                    if self.cols:
                        cols = self.cols.copy()
                    else:
                        cols = list(df_raw.columns)
                        if 'date' in cols:
                            cols.remove('date')
                    if 'date' in df_raw.columns:
                        df_raw = df_raw[['date'] + cols]
                    else:
                        df_raw = df_raw[cols]
                else:
                    if self.cols:
                        cols = self.cols.copy()
                        if self.target in cols:
                            cols.remove(self.target)
                    else:
                        cols = list(df_raw.columns)
                        if self.target in cols:
                            cols.remove(self.target)
                        if 'date' in cols:
                            cols.remove('date')
                    df_raw = df_raw[['date'] + cols + [self.target]]

                num_train = int(len(df_raw) * 0.7)
                num_test = int(len(df_raw) * 0.2)
                num_vali = len(df_raw) - num_train - num_test
                border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
                border2s = [num_train, num_train + num_vali, len(df_raw)]
                border1 = border1s[self.set_type]
                border2 = border2s[self.set_type]

                if self.features in ['M', 'MS']:
                    cols_data = df_raw.columns[1:]
                    df_data = df_raw[cols_data]
                else:
                    df_data = df_raw[[self.target]]

                if self.scale:
                    train_data = df_data[border1s[0]:border2s[0]]
                    scaler_tmp = StandardScaler()
                    scaler_tmp.fit(train_data.values)
                    self.scaler.append(scaler_tmp)
                    data = scaler_tmp.transform(df_data.values)
                else:
                    data = df_data.values

                df_stamp = df_raw[['date']][border1:border2].copy()
                df_stamp['date'] = pd.to_datetime(df_stamp.date)
                data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
                df_stamp['year'] = df_stamp['date'].dt.year
                df_stamp['month'] = df_stamp['date'].dt.month
                df_stamp['day'] = df_stamp['date'].dt.day

                data_x.append(data[border1:border2])
                if self.inverse:
                    data_y.append(df_data.values[border1:border2])
                else:
                    data_y.append(data[border1:border2])

                self.data_stamp = data_stamp
                self.data_time = df_stamp[['year', 'month', 'day']].values

            min_split_rows = min(arr.shape[0] for arr in data_x)
            if any(arr.shape[0] != min_split_rows for arr in data_x):
                data_x = [arr[:min_split_rows] for arr in data_x]
                data_y = [arr[:min_split_rows] for arr in data_y]
                self.data_stamp = self.data_stamp[:min_split_rows]
                self.data_time = self.data_time[:min_split_rows]
            self.data_x = np.stack(data_x, axis=2)
            self.data_y = np.stack(data_y, axis=2)
        else:
            filepath = os.path.join(self.root_path, self.data_path)
            df_raw = _read_table(filepath)

            # Normalize time column: rename 'time' → 'date' for internal consistency
            time_col = _detect_time_col(df_raw)
            if time_col is None:
                # Time is in the index — promote to a 'date' column
                df_raw = df_raw.reset_index()
                if df_raw.columns[0] != 'date':
                    df_raw = df_raw.rename(columns={df_raw.columns[0]: 'date'})
            elif time_col != 'date':
                df_raw = df_raw.rename(columns={time_col: 'date'})

            if self.features == 'M':
                if self.cols:
                    cols = self.cols.copy()
                else:
                    cols = list(df_raw.columns)
                    cols.remove('date')
                df_raw = df_raw[['date'] + cols]
            else:
                if self.cols:
                    cols = self.cols.copy()
                    cols.remove(self.target)
                else:
                    cols = list(df_raw.columns)
                    cols.remove(self.target)
                    cols.remove('date')
                df_raw = df_raw[['date'] + cols + [self.target]]

            num_train = int(len(df_raw) * 0.7)
            num_test = int(len(df_raw) * 0.2)
            num_vali = len(df_raw) - num_train - num_test
            border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
            border2s = [num_train, num_train + num_vali, len(df_raw)]
            border1 = border1s[self.set_type]
            border2 = border2s[self.set_type]

            if self.features in ['M', 'MS']:
                cols_data = df_raw.columns[1:]
                df_data = df_raw[cols_data]
            else:
                df_data = df_raw[[self.target]]

            if self.scale:
                train_data = df_data[border1s[0]:border2s[0]]
                self.scaler.fit(train_data.values)
                data = self.scaler.transform(df_data.values)
            else:
                data = df_data.values

            df_stamp = df_raw[['date']][border1:border2].copy()
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
            df_stamp['year'] = df_stamp['date'].dt.year
            df_stamp['month'] = df_stamp['date'].dt.month
            df_stamp['day'] = df_stamp['date'].dt.day
            self.data_time = df_stamp[['year', 'month', 'day']].values

            self.data_x = data[border1:border2]
            if self.inverse:
                self.data_y = df_data.values[border1:border2]
            else:
                self.data_y = data[border1:border2]
            self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        seq_y_time = self.data_time[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark, seq_y_time

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        if isinstance(self.scaler, list):
            num = data.shape[-1]
            for i in range(num):
                data[:, :, :, i] = self.scaler[i].inverse_transform(data[:, :, :, i])
            return data
        return self.scaler.inverse_transform(data)


class Dataset_ETT_hour(Dataset_Custom):
    pass


class Dataset_ETT_minute(Dataset_Custom):
    pass


class Dataset_Pred(Dataset_Custom):
    pass
