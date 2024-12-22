import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score as r2

def model_eval(model = None,last = None, X = None, y = None, u = None, l = None):
  '''
  rmse, mae, mape, CC 평가결과
  입력값
  model,
  last = True(seq2seq) or False(ow) 끝값 평가 유무!
  X = [X_train, X_val, X_test], 
  y = [y_train, y_val, y_test], 
  u, l
  '''
  X_train, X_val, X_test = X
  y_train, y_val, y_test = y

  eval_df = pd.DataFrame()
  for i in ['RMSE', 'MAE', 'MAPE(%)', 'CC']:
    eval_df[i] = []
  
  for X, y, t in zip([X_train, X_val, X_test], [y_train, y_val, y_test], ['Train', 'Validation', 'Test']):
    pred = model.predict(X)
    
    if last == False:
      pred = pred * (u - l) + l
      y = y.reshape(y.shape[0], -1) * (u - l) + l
    elif last == True:
      pred = pred.reshape(y.shape[0], -1)[:,-1] * (u - l) + l
      y = y.reshape(y.shape[0], -1)[:,-1] * (u - l) + l

    eval_df.loc[t] = [f'{mean_squared_error(pred, y)**0.5:.2f}',
                      f'{mean_absolute_error(pred, y):.2f}',
                      f'{mean_absolute_percentage_error(pred, y)*100:.2f}',
                      f'{np.corrcoef(pred.reshape(-1), y.reshape(-1))[0,1]:.2f}']
  return eval_df