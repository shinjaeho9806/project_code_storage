import matplotlib.pyplot as plt

def train_loss_plot(history = None, model_name = None, epochs = None, save_path = None):
  '''
  학습 loss 그래프
  history = None, model_name = None, 
  epochs = None, save_path = None
  '''
  plt.plot(history.history['loss'], label = 'train loss')
  plt.plot(history.history['val_loss'], label = 'valid loss')
  plt.title(f'{model_name}')
  plt.xlabel('epochs')
  plt.ylabel('loss', rotation = 360, labelpad = 20)
  plt.xticks([i for i in range(epochs)],[i+1 for i in range(epochs)])
  plt.legend(loc = 'upper right')
  plt.savefig(f'{save_path}')

def predict_plot(model = None, last = None, X = None, y = None, u = None, l = None, title = None, save_path = None):
  '''
  모델 예측 plotting함수
  model = None,
  last = True(seq2seq) or False(OW)
  X = [], y = [], 
  u = None, l = None, 
  title = None, 
  save_path = None
  '''
  plt.figure(figsize = (15,9))
  for i, tvt in enumerate(['train', 'valid', 'test']):
    if last == False:
      tmp_X = model.predict(X[i]) * (u - l) + l
      tmp_y = y[i].reshape(y[i].shape[0], -1)  * (u - l) + l
    elif last == True:
      tmp_X = model.predict(X[i])[:,-1] * (u - l) + l
      tmp_y = y[i].reshape(y[i].shape[0], -1)[:,-1]  * (u - l) + l
    
    plt.subplot(3,1,i+1)
    plt.plot(tmp_X, label = 'predict')
    plt.plot(tmp_y, label = 'original')
    plt.legend(loc = 'upper right')
    plt.title(f'{tvt}')
    plt.ylim(50,380)
  plt.suptitle(f'{title}')
  plt.savefig(f'{save_path}')