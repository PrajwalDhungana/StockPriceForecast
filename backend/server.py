from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
app.url_map.strict_slashes = False
CORS(app)

@app.route('/items')
def get_items():
    return {"items": ["item1", "item2", "item3"]}

@app.route('/submit', methods=['POST'])
def submit():
    data = request.get_json()
    tickerSymbol = data['tickerSymbol']
    print(tickerSymbol)
    return jsonify({'tickerSymbol': tickerSymbol})

if __name__ == '__main__':
    app.run(debug=True)