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
from prediction_model import predict

app = Flask(__name__)
app.url_map.strict_slashes = False
CORS(app)


@app.route('/status')
def get_status():
    return {"status": ["Online"]}


@app.route('/submit', methods=['POST'])
def submit():
    try:
        data = request.get_json()
        tickerSymbol = data['tickerSymbol']
        print(tickerSymbol)

        prediction = predict(tickerSymbol)

        return jsonify(data=prediction)
    except Exception as ex:
        return jsonify(ex)

if __name__ == '__main__':
    app.run(debug=True)
