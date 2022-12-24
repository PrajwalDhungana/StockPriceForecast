from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

app = Flask(__name__)
app.url_map.strict_slashes = False
CORS(app)

@app.after_request
def set_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "*"
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/items')
def get_items():
    return {"items": ["item1", "item2", "item3"]}

@app.route('/insertTicker', methods=['POST', 'GET'])
def insertTicker():
    if request.methods == 'POST':
        tickerSymbol = request.json['tickerSymbol']
        return tickerSymbol
    else:
        return render_template('')

@app.route("/<ticker>")
def ticker(ticker):
    return f"<h1>ticker</h1>"

if __name__ == '__main__':
    app.run(debug=True)