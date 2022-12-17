from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


@app.route('/items')
def get_items():
    return {"items": ["item1", "item2", "item3"]}


if __name__ == '__main__':
    app.run(debug=True)
