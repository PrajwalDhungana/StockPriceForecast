from flask import Flask, request, jsonify
from flask_cors import CORS
from prediction_model import predict

app = Flask(__name__)
app.url_map.strict_slashes = False
CORS(app)


@app.route('/status')
def get_status():
    return {"status": ["Online"]}


@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    tickerSymbol = data['tickerSymbol']
    prediction = predict(tickerSymbol)
    # return {'prediction': prediction}
    return jsonify(data=prediction)

if __name__ == '__main__':
    app.run(debug=True)