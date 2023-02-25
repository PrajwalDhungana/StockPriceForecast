import keys as ky
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import math
import numpy as np
from pandas_datareader import data as pdr
import datetime
from datetime import date
import os
import yfinance as yf
yf.pdr_override()


def fetch_data(ticker):
    # # Fetch the data from tiingo API
    # data = pdr.get_data_tiingo(ticker, api_key=ky.TIINGO_API_KEY)
    # df = pd.DataFrame(data=data)
    # return data, df

    # We can get data by our choice by giving days bracket
    start_date = "2017-01-01"
    today = date.today()
    print(today)

    files = []

    def getData(ticker):
        print(ticker)
        data = pdr.get_data_yahoo(ticker, start=start_date, end=today)
        dataname = ticker
        files.append(dataname)
        SaveData(data, dataname)

    # Create a data folder in your current dir.
    def SaveData(df, filename):
        if not (os.path.isdir('data')):
            os.mkdir('data')
        df.to_csv('./data/'+filename+'.csv')

    # This will pass ticker to get data, and save that data as file.
    getData(ticker)

def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]
        dataX.append(a)
        dataY.append(dataset[i+time_step, 0])
    return np.array(dataX), np.array(dataY)


def predict(tickerSymbol):
    fetch_data(tickerSymbol)
    data = pd.read_csv('./data/' + tickerSymbol + '.csv')
    print(data.tail())
    df = data.reset_index()

    close = df['Adj Close'].tolist()
    date = pd.to_datetime(df['Date']).map(lambda x: str(x.date())).tolist()
    df1 = close

    scaler = MinMaxScaler(feature_range=(0, 1))
    df1 = scaler.fit_transform(np.array(df1).reshape(-1, 1))

    # Train the data
    train_percent = 0.65
    training_size = int(len(close)*train_percent)
    testing_size = len(df1) - training_size

    train_data, test_data = df1[0:training_size], df1[training_size:len(df1)]
    train_data_date, test_data_date = date[0:training_size], date[training_size:len(
        date)]
    time_step = 100
    X_train, Y_train = create_dataset(train_data, time_step)
    X_test, Y_test = create_dataset(test_data, time_step)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    # Fit the trained data to the LSTM Model and predict the next 30 days
    list_output = []
    model = Sequential()
    hidden_layer = 50
    model.add(LSTM(hidden_layer, return_sequences=True,
              input_shape=(X_train.shape[1], X_train.shape[2])))
    model.add(LSTM(hidden_layer, return_sequences=True))
    model.add(LSTM(hidden_layer))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(X_train, Y_train, validation_data=(
        X_test, Y_test), epochs=1, batch_size=64, verbose=1)

    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)
    train_error = math.sqrt(mean_squared_error(Y_train, train_predict))
    test_error = math.sqrt(mean_squared_error(Y_test, test_predict))

    # Predict the next 7 days
    x_input = test_data[(len(test_data) - 100):].reshape(1, -1)
    temp_input = list(x_input)
    temp_input = temp_input[0].tolist()
    n_steps = 100
    i = 0
    days = 30
    while (i < days):
        if (len(temp_input) > 100):
            x_input = np.array(temp_input[1:])
            x_input = x_input.reshape(1, -1)
            x_input = x_input.reshape(1, n_steps, 1)
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            temp_input = temp_input[1:]
            list_output.extend(yhat.tolist())
            i += 1
        else:
            x_input = x_input.reshape(1, n_steps, 1)
            yhat = model.predict(x_input, verbose=0)
            temp_input.extend(yhat[0].tolist())
            list_output.extend(yhat.tolist())
            i += 1

    # Generate next prediction date
    i=1
    pred_date = []
    while(i <= days) :
        new_dates = datetime.date.today() + datetime.timedelta(days=i)
        pred_date.append(new_dates.strftime ('%Y-%m-%d'))
        i += 1

    print("Next prediction dates : ", pred_date)

    day_new = np.arange(1, 101)
    day_pred = np.arange(101, 101 + days)

    print("Day_NEW : ", day_new)
    print("Day_Pred : ", day_pred)

    # Plot the predicted values
    look_back = 100
    trainPredictPlot = np.empty_like(df1)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[look_back:len(
        train_predict) + look_back, :] = train_predict
    testPredictPlot = np.empty_like(df1)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(train_predict) + (look_back*2) +
                    1: len(df1) - 1, :] = test_predict
    df2 = df1.tolist()
    df2.extend(list_output)
    train_price = train_predict.tolist()
    test_price = test_predict.tolist()
    
    # Transformed output
    pred_output = scaler.inverse_transform(list_output).tolist()
    pred_close = [item for subarr in pred_output for item in subarr]
    transformed_df1 = scaler.inverse_transform(df1).tolist()

    print("TRANSFORMED LIST OUTPUT : ", pred_close)

    # Object that contains normal trends data
    keys1 = ['date', 'close']
    values = [date, close]
    trends = {
        key: value for key,
        value in zip(keys1, values)
    }

    #Object that contains prediction data
    keys2 = ['date', 'close']
    values2 = [pred_date, pred_close]
    predicts = {
        key: value for key,
        value in zip(keys2, values2)
    }

    return trends, predicts