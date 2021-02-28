#!/usr/bin/env python
import logging

from flask import Flask
from flask import render_template
from models.engine import *

app = Flask(__name__,
            static_url_path='', 
            static_folder='static',
            template_folder='templates')

@app.route('/')
def hello():
    return render_template('index.html')

@app.route('/recommend_stocks', methods=['GET'])
def recommend_stocks():
    print('..enterprise has engaged..')
    selected_stocks = run_engine(n_tickers=10)
    print(selected_stocks)
    return selected_stocks

@app.errorhandler(500)
def server_error(e):
    # Log the error and stacktrace.
    logging.exception('An error occurred during a request.')
    return 'An internal error occurred.', 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)