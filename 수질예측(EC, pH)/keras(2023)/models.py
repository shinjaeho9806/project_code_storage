import keras
from keras.layers import Input, Dense, RepeatVector, LSTM, GRU, TimeDistributed, Bidirectional
from keras.models import Model, Sequential

def single_RNN(model_type = 'lstm', unit = None, window = None):
    '''
    단방향 RNN
    model_type : lstm, gru, bi_lstm, bi_gru
    unit : RNN의 노드수
    window : 입력 윈도우 길이
    '''
    model = Sequential()
    model.add(Input(batch_shape=(None, window, 1)))
    if model_type == "lstm":
        model.add(LSTM(unit, return_sequences = True))
        model.add(LSTM(unit))
    elif model_type == "gru":
        model.add(GRU(unit, return_sequences = True))
        model.add(GRU(unit))
    elif model_type == "bi_lstm":
        model.add(Bidirectional(LSTM(unit)))
    elif model_type == "bi_gru":
        model.add(Bidirectional(GRU(unit)))
    model.add(Dense(1))
    return model

def RNN_seq2seq(model_type = 'lstm', unit = None, window = None, interval = None):
    '''
    RNN_seq2seq
    model_type : lstm, gru
    unit : RNN의 노드수
    window : 입력 윈도우 길이
    interval : 예측 구간 길이
    '''
    model = Sequential()
    model.add(Input(batch_shape=(None, window, 1)))
    if model_type == 'lstm':
        model.add(LSTM(unit))
        model.add(RepeatVector(interval))
        model.add(LSTM(unit, return_sequences = True))
    elif model_type == 'gru':
        model.add(GRU(unit))
        model.add(RepeatVector(interval))
        model.add(GRU(unit, return_sequences = True))
    model.add(TimeDistributed(Dense(1)))
    return model