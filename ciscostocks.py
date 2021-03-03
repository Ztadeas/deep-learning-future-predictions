import pandas as pd 
import numpy as np
from keras import models
from keras import layers
from keras import optimizers

dir_path = "C:\\Users\\Tadeas\\Downloads\\cisco\\CSCO_2006-01-01_to_2018-01-01.csv"

everything = pd.read_csv(dir_path)

close = []

valid = 2600

for i in range(3019):
  close.append(everything["Close"][i])

close = np.asarray(close, dtype="float32")

close = np.reshape(close, (3019, 1))

def preproces(data):
  mean = data.mean(axis = 0)
  data -= mean
  std = data.std(axis = 0)
  data = data / std + 2
  
  return data

def to_train_val(validation_split, data):
  train = data[:validation_split]
  val = data[validation_split:]
  
  return train, val

data = preproces(close)


train_data, val_data = to_train_val(valid, data)

def to_timeseries(data, lookback, future):
  features = []
  labels = []
  for i in range(len(data) - lookback - 1):
    features.append(data[i: (i+ lookback), 0])
    labels.append(data[(i+lookback), 0])

  return features, labels

x_train, y_train = to_timeseries(train_data, 5, 8)
x_val, y_val = to_timeseries(val_data, 5, 8)


x_train = np.asarray(x_train, dtype= "float32").reshape(2594, 5, 1)
y_train = np.asarray(y_train, dtype= "float32").reshape(2594, 1)

x_val = np.asarray(x_val, dtype="float32").reshape(413, 5, 1)
y_val = np.asarray(y_val, dtype="float32").reshape(413, 1)

m = models.Sequential()

m.add(layers.LSTM(64, return_sequences=True, dropout=0.1, recurrent_dropout=0.5, input_shape= (5, 1)))
m.add(layers.LSTM(64, activation="relu", dropout= 0.1, recurrent_dropout= 0.5))
m.add(layers.Dense(1))

m.compile(optimizer=optimizers.Adam(lr=0.001), loss="mse", metrics= ["mae"])

m.fit(x_train, y_train, epochs= 30, batch_size= 16, validation_data=(x_val, y_val))







    
    
  