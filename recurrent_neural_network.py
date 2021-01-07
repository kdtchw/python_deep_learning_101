# Data preprocessing

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_train = pd.read_csv("Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:, 1:2].values

dataset_test = pd.read_csv("Google_Stock_Price_Test.csv")
test_set = dataset_test.iloc[:, 1:2].values

from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0,1))
training_set_scaled = sc.fit_transform(training_set)

X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# creating a data structure with 60 timesteps and 1 output
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))


# Building RNN

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dropout

# initializing the rnn
regressor = Sequential()

# adding the 1st LSTM layer and some dropout regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# adding the 2nd LSTM layer and some dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# adding the 3rd LSTM layer and some dropout regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

# adding the 4th LSTM layer and some dropout regularization
regressor.add(LSTM(units=50))
regressor.add(Dropout(0.2))

# adding the output layer
regressor.add(Dense(units=1))

# compiling the RNN
regressor.compile(optimizer="adam", loss="mean_squared_error")

# fitting the RNN to training set
regressor.fit(X_train, y_train, epochs=100, batch_size=32)


# Prediction and visualization

dataset_total = pd.concat((dataset_train["Open"], dataset_test["Open"]), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1, 1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(test_set, color="red", label="Real Google Stock Price")
plt.plot(predicted_stock_price, color="blue", label="Predicted Google Stock Price")
plt.title("Google Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Google Stock Price")
plt.legend()
plt.show()
