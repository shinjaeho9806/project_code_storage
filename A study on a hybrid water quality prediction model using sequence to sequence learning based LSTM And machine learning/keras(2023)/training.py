import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
import math

def train_model(model = None, save_path = None, optim = None, X = None, y = None, lr = 0.001, batch_size = 128, epochs = 5):
  '''
  모델 학습함수
  model = model,
  save_path = 저장경로,
  optim = 사용할 옵티마이저 명,
  lr = 학습률,
  batch_size = 배치크기
  epochs = 에포크 수
  '''
  X_train, X_val = X[0], X[1]
  y_train, y_val = y[0], y[1]
  
  model_checkpoint = ModelCheckpoint(monitor = 'val_loss', save_best_only = True, verbose=1, filepath = f'{save_path}')
  #model_earlystopping = EarlyStopping(monitor="val_loss", patience=20)
  
  if optim == "Adam":
    optimizer = keras.optimizers.Adam(learning_rate = lr)
  elif optim == "AdamW":
    optimizer = keras.optimizers.AdamW(learning_rate = lr)
  elif optim == "RMSprop":
    optimizer = keras.optimizers.RMSprop(leraning_rate = lr)

  model.compile(
      optimizer = optimizer,
      loss = keras.losses.mean_squared_error)
  
  train_generator = DataGenerator(
      batch_size = batch_size,
      X= X_train,
      y = y_train)
  
  val_generator = DataGenerator(
      batch_size = batch_size,
      X = X_val,
      y = y_val)
  
  history = model.fit(train_generator, validation_data = val_generator, epochs = epochs,
                    verbose = 1, callbacks = [model_checkpoint])
  return history


class DataGenerator(keras.utils.Sequence):
    '''
    배치단위 데이터 로더
    batch_size, X, y 입력!
    '''
    def __init__(self , batch_size , X , y):
        self.batch_size = batch_size
        self.X = X
        self.y = y

    def __len__(self):
        return math.ceil(len(self.y) / self.batch_size)

    def __getitem__(self , idx):
        start, end = idx * self.batch_size, (idx + 1) * self.batch_size
        return self.X[start:end], self.y[start:end]