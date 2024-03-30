import numpy as np
import pandas as pd
import math
import keras

def EC_pH(path = '', type = ''):
    '''
    데이터 분할 및 스케일링 함수
    path : 엑셀파일 경로
    type : EC or pH
    반환값
    train_data, valid_data, test_data, upper, lower
    '''
    data = pd.read_excel(path, sheet_name = type)
    data = data[f'{type}_origin']
    data = np.array(data).reshape(-1,1)

    temp = int(round(data.shape[0]/5,0))
    train_data = data[:-temp*2, :]
    val_data = data[-temp*2:-temp, :]
    test_data = data[-temp:,:]

    u, l = train_data.max(), train_data.min()
    train_data, val_data, test_data = (train_data - l)/(u-l), (val_data - l)/(u-l), (test_data - l)/(u-l)
    return train_data, val_data, test_data, u, l

def get_one_sequential(data, window = 24, interval = 48):
  '''
  single_rnn과 bi_rnn용 데이터 생성 함수
  data : 데이터
  * X = (sample_size, window_size, feature_size)
  * y = (sample_size, 1, feature_size)
  반환값 
  X, y
  '''
  sample_size = data.shape[0]-(window+interval)+1
  X, y = np.zeros((sample_size, window, 1)), np.zeros((sample_size, 1, 1))
  X[:], y[:] = np.nan, np.nan
  for i in range(sample_size):
    X[i], y[i] = data[i:i+window], data[i+window+interval-1]
  return X, y

def get_multi_sequential(data, window = 24, interval = 48):
  '''
  rnn_seq2seq용 데이터 생성 함수
  data : 데이터
  * X = (sample_size, window_size, feature_size)
  * y = (sample_size, 1, feature_size)
  반환값 : X, y
  '''
  sample_size = data.shape[0]-(window+interval)+1
  X, y = np.zeros((sample_size, window, 1)), np.zeros((sample_size, 1, 1))
  X[:], y[:] = np.nan, np.nan
  for i in range(sample_size):
    X[i], y[i] = data[i:i+window], data[i+window+interval-1]
  return X, y