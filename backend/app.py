from flask import Flask, request, jsonify
from flask_cors import CORS

import numpy as np
import pandas as pd
import pandas_datareader as pdr
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
import math
import keys as ky

app = Flask(__name__)
app.url_map.strict_slashes = False
CORS(app)


@app.route('/status')
def get_status():
    return {"status": ["Online"]}


@app.route('/submit', methods=['POST'])
def submit():
    try:
        # 1. Collect the stock data
        # 2. Preprocess the Data - Train & Test
        # 3. Create an Stacked LSTM Model
        # 4. Predict the test data and plot the output
        # 5. Predict the future 30 dats and plot the output

        data = request.get_json()
        tickerSymbol = data['tickerSymbol']
        print(tickerSymbol)

        # Fetch the data from tiingo API
        data = pdr.get_data_tiingo(tickerSymbol, api_key=ky.tiingo_api_key)
        df = pd.DataFrame(data=data)
        # df.to_csv(tickerSym+'.csv')

        data = data.reset_index()
        df = pd.DataFrame()
        df['Symbol'] = data['symbol']
        df['Date'] = data['date']
        df['Open'] = data['open']
        df['High'] = data['high']
        df['Low'] = data['low']
        df['Close'] = data['close']
        df['AdjClose'] = data['adjClose']
        df['Volume'] = data['volume']
        df.to_csv(tickerSymbol+'.csv', index=False)
        pd.read_csv(tickerSymbol+'.csv')

        df_date_min = pd.to_datetime(df['Date']).map(lambda x: str(x.date()))
        # close = df['Close']
        # date = df['Date'].tolist()
        # df1 = close

        df_close = df['Close'].tolist()
        df_date = df_date_min.tolist()
        df1 = df_close

        datas = [df_close, df_date]
        datas

        scaler = MinMaxScaler(feature_range=(0, 1))
        df_close_scaled = scaler.fit_transform(np.array(df_close).reshape(-1, 1))

        def create_dataset(dataset, time_step=1):
            dataX, dataY = [], []
            for i in range(len(dataset)-time_step-1):
                a = dataset[i:(i+time_step), 0]
                dataX.append(a)
                dataY.append(dataset[i+time_step, 0])
            return np.array(dataX), np.array(dataY)

        # Train the data
        train_percent = 0.65
        training_size = int(len(df_close)*train_percent)
        training_date_size = int(len(df_date_min)*train_percent)

        testing_size = len(df_close) - training_size

        train_data, test_data = df_close[0:training_size], df_close[training_size:len(df_close)]
        train_data_date, test_data_date = df_date_min[0:training_date_size],df_date_min[training_date_size:len(df_date_min)]

        time_step = 100
        X_train, Y_train = create_dataset(train_data, time_step)
        X_test, Y_test = create_dataset(test_data, time_step)

        X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
        X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

        # Fit the trained data to the LSTM Model
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
            X_test, Y_test), epochs=2, batch_size=64, verbose=1)
        train_predict = model.predict(X_train)
        test_predict = model.predict(X_test)
        train_predict = scaler.inverse_transform(train_predict)
        test_predict = scaler.inverse_transform(test_predict)
        train_error = math.sqrt(mean_squared_error(Y_train, train_predict))
        test_error = math.sqrt(mean_squared_error(Y_test, test_predict))

        # Predict the next 30 days
        x_input = test_data[(len(test_data) - 100):].reshape(1, -1)
        temp_input = list(x_input)
        temp_input = temp_input[0].tolist()
        n_steps = 100
        i = 0
        while (i < 30):
            if (len(temp_input) > 100):
                x_input = np.array(temp_input[1:])
                print("{} day input {}".format(i, x_input))
                x_input = x_input.reshape(1, -1)
                x_input = x_input.reshape(1, n_steps, 1)
                yhat = model.predict(x_input, verbose=0)
                print("{} day output {}".format(i, yhat))
                temp_input.extend(yhat[0].tolist())
                temp_input = temp_input[1:]
                list_output.extend(yhat.tolist())
                i += 1
            else:
                x_input = x_input.reshape(1, n_steps, 1)
                yhat = model.predict(x_input, verbose=0)
                print(yhat[0])
                temp_input.extend(yhat[0].tolist())
                print(len(temp_input))
                list_output.extend(yhat.tolist())
                print(list_output)
                i += 1

        day_new = np.arange(1, 101)
        day_pred = np.arange(101, 131)

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
        transformed_list_output = scaler.inverse_transform(list_output).tolist()
        transformed_df1 = scaler.inverse_transform(df1).tolist()

        keys = ['date', 'close']
        values = [date, close]
        result = {
            key: value for key,
            value in zip(keys, values)
        }
        
        return jsonify(data=result)
    except Exception:
        return jsonify({'Status': 'error'})

if __name__ == '__main__':
    app.run(debug=True)
