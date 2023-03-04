# from flask import Flask, request, jsonify
# from flask_cors import CORS

# from prediction_model import predict

# app = Flask(__name__)
# app.url_map.strict_slashes = False
# CORS(app)


# @app.route('/status')
# def get_status():
#     return {"status": ["Online"]}


# @app.route('/submit', methods=['POST'])
# def submit():
#     try:
#         data = request.get_json()
#         tickerSymbol = data['tickerSymbol']
#         print(tickerSymbol)

#         prediction = predict(tickerSymbol)

#         return jsonify(data=prediction)
#     except Exception as ex:
#         return jsonify(ex)

# if __name__ == '__main__':
#     app.run(debug=True)

from flask import Flask, request
from flask_cors import CORS
import datetime as dt
import sys
from prediction_model import lstm_predict

app = Flask(__name__)
app.url_map.strict_slashes = False
CORS(app)
app.config['TEMPLATES_AUTO_RELOAD'] = True
# prevent caching so website can be updated dynamically
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
    
# app routes are urls which facilitate
# data transmit, mainly:
# Get and Post requests
@app.route('/status')
def get_status():
    return {"status": ["Online"]}

@app.route('/submit', methods=['POST'])
def submit():
    # POST request
    if request.method == 'POST':
        # convert to JSON
        data = request.get_json(force=True)
        tickerSymbol = data['tickerSymbol'].upper()
        # start = data["startDate"].split("-")
        # end = data["endDate"].split("-")
        start = ['2020', '01', '01']
        end = ['2023', '03', '15']

        # convert strings to integers
        start, end = [int(s) for s in start], [int(s) for s in end]

        try:
            # get original stock data, train and test results
            actual, trend_dates, train_res, test_res, pred_res, accuracy, num_days = lstm_predict(tickerSymbol, start, end)
        except:
            # error info
            e = sys.exc_info()
            print(e)
            print("handle_nn fail")
            return "error", 404

        # convert pandas dataframe to list
        actual = [i for i in actual]
        train_res, test_res, pred_res = [i for i in train_res[0][:]], [i for i in test_res[0][:]], [i for i in pred_res[0][:]]
        train_date, test_date, pred_date = [i for i in trend_dates[6:len(train_res)]], [i for i in trend_dates[len(train_res)+6:]], [i for i in trend_dates[len(pred_res)+num_days]]

        # Generate next prediction date
        i=1
        pred_date = []
        while(i <= num_days) :
            new_dates = dt.datetime.today() + dt.timedelta(days=i)
            pred_date.append(new_dates.strftime ('%Y-%m-%d'))
            i += 1

        print("Next prediction dates : ", pred_date)
        pred_res = pred_res[len(pred_res)-num_days:]
        print("Next prediction prices : ", pred_res)

        actualX = trend_dates
        trainX = train_date
        testX = test_date
        predX = pred_date

        # connect training and test lines in plot
        test_res.insert(0, train_res[-1])
        testX.insert(0, trainX[-1])

        # connect training and test lines in plot
        pred_res.insert(0, test_res[-1])
        predX.insert(0, testX[-1])

        return {"stock" : tickerSymbol,
                "actual" : actual,
                "actualX" : actualX,
                "train" : train_res,
                "trainX" : trainX, 
                "test" : test_res,
                "testX" : testX,
                "pred" : pred_res,
                "predX" : predX,
                "accuracy" : accuracy}
        
if __name__ == '__main__':
    app.run(debug=True)