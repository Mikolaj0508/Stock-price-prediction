import math
import pandas_datareader as web
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
!pip install --upgrade pandas
!pip install --upgrade pandas-datareader

#importing data
df = web.DataReader('INTC', data_source='yahoo', start='2012-01-01', end='2019-12-17')
print(df)
df.shape

#visualization of close price data
plt.figure(figsize=(16,8))
plt.title("Close price history")
plt.plot(df['Close'])
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price USD', fontsize=18)
plt.show()

#creating new data frame and converting it into numpy array
data = df.filter(['Close'])
dataset = data.values
#number of rows for training model
training_data_len = math.ceil(len(dataset)*0.8)
print(training_data_len)

#scale the data
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)
print(scaled_data)

#training data set and scaled data set
train_data = scaled_data[0:training_data_len, :]
#split the data into x_train and y_train
x_train = []
y_train = []

for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0])
  y_train.append(train_data[i, 0])
  if i <= 61:
    print(x_train)
    print(y_train)
    print()

#converting the x_train and y_train to numpy array
x_train, y_train = np.array(x_train), np.array(y_train)

#reshape the data
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
x_train.shape

#build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape = (x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

#compile model
model.compile(optimizer='adam', loss='mean_squared_error')

#training the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

#Create testing data set
test_data = scaled_data[training_data_len-60: , :]
x_test = []
y_test = dataset[training_data_len:, :]
for i in range(60, len(test_data)):
  x_test.append(test_data[i-60:i, 0])
  
#coverting the data to numpy array
x_test = np.array(x_test)

#reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

#model predicting price values
predictions = model.predict(x_test)
predictions = scaler.inverse_transform(predictions)

#root mean squared error
rmse = np.sqrt( np.mean(predictions - y_test)**2)
print(rmse)

#plot the data
train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions

plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close price USD', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc='lower right')
plt.show()

#show the valid and predicted price
print(valid)

intel_quote = web.DataReader('INTC', data_source='yahoo', start='2012-01-01', end='2019-12-17')
#new data frame
new_df = intel_quote.filter(['Close'])
#get the last 60 days closing price and convert dataframe to array
last_60_days = new_df[-60:].values
#scale the data to be values between 0 and 1
last_60_days_scaled = scaler.transform(last_60_days)

X_test = []
X_test.append(last_60_days_scaled)
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

intel_quote2 = web.DataReader('INTC', data_source='yahoo', start='2019-12-18', end='2019-12-18')
print(intel_quote2['Close'])
