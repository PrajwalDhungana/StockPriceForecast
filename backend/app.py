import numpy as np
import pandas as pd
import pandas_datareader as pdr
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import math

### 1. Collect the stock data
### 2. Preprocess the Data - Train & Test
### 3. Create an Stacked LSTM Model
### 4. Predict the test data and plot the output
### 5. Predict the future 30 dats and plot the output
def get_data(tickerSymbol):
    key='c1755d754ae022eef0eb4bf39ba4c92a9216406a'
    df = pdr.get_data_tiingo(tickerSymbol, api_key=key)
    df.to_csv(tickerSymbol+'.csv')


def read_data(tickerSymbol):
    df = pd.read_csv(tickerSymbol+'.csv')
    df1 = df.reset_index()['close']
    plt.plot(df1)
    scaler = MinMaxScaler(feature_range=(0,1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))
    return df1, scaler

df1, scaler = read_data(tickerSymbol="TSLA")

def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i+time_step, 0])
    return np.array(dataX), np.array(dataY)

def train_test():
    training_size = int(len(df1)*0.65)

    testing_size = len(df1) - training_size
    train_data, test_data = df1[0:training_size], df1[training_size:len(df1)]
    time_step = 100
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)

    X_train=X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test=X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    return np.array(train_data), np.array(test_data), np.array(X_train), np.array(X_test), np.array(Y_train), np.array(Y_test)
    # return X_train, X_test

train_data, test_data, X_train, X_test, Y_train, Y_test = train_test()
# X_train, X_test = train_test()

def stacked_LSTM():
    list_output = []
    model = Sequential()
    hidden_layer = 50
    model.add(LSTM(hidden_layer, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(hidden_layer, return_sequences=True))
    model.add(LSTM(hidden_layer))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), epochs=10, batch_size=64, verbose=1)
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    train_error = math.sqrt(mean_squared_error(Y_train, train_predict))
    test_error = math.sqrt(mean_squared_error(Y_test, test_predict))

    x_input = test_data[(len(test_data) - 100):].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    n_steps = 100
    i=0
    while(i<30):
        if(len(temp_input)>100):
            x_input=np.array(temp_input[1:])
            print("{} day input {}".format(i, x_input))
            x_input=x_input.reshape(1, -1)
            x_input=x_input.reshape(1, n_steps, 1)
            yhat=model.predict(x_input, verbose=0)
            print("{} day output {}".format(i, yhat))
            temp_input.extend(yhat[0].tolist())
            temp_input=temp_input[1:]
            list_output.extend(yhat.tolist())
            i += 1
        else:
            x_input=x_input.reshape(1, n_steps, 1)
            yhat=model.predict(x_input, verbose=0)
            print(yhat[0])
            temp_input.extend(yhat[0].tolist())
            print(len(temp_input))
            list_output.extend(yhat.tolist())
            print(list_output)
            i += 1
        
    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 131)

    return np.array(train_predict), np.array(test_predict), np.array(train_error), np.array(test_error), np.array(list_output), np.array(day_new), np.array(day_pred)

train_predict, test_predict, train_error, test_error, list_output, day_new, day_pred = stacked_LSTM()

def plot_values():
    look_back = 100
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:,:] = np.nan
    trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:,:] = np.nan
    testPredictPlot[len(train_predict) + (look_back*2) + 1 : len(df1) - 1, :] = test_predict
    plt.plot(scaler.inverse_transform(df1))
    plt.plot(trainPredictPlot)
    plt.plot(testPredictPlot)
    plt.show()
    plt.plot(day_new, scaler.inverse_transform(df1[(len(df1) - 100):]))
    plt.plot(day_pred, scaler.inverse_transform(list_output))
    plt.show()
    df2 = df1.tolist()
    df2.extend(list_output)
    plt.plot(df2[1000:])
    plt.show()

def predict_stock(tickerSymbol):
    get_data(tickerSymbol)
    train_test()
    stacked_LSTM()
    plot_values()
    print("done!!!!")

predict_stock(tickerSymbol='TSLA')