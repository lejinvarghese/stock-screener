#!/usr/bin/env python
import logging
import json

from flask import Flask
from flask import render_template
from core.analysis import run as analyze
from core.optimizer import run as optimize

app = Flask(__name__,
            static_url_path='',
            static_folder='static')


@app.route('/')
def template():
    """
    Returns base template
    """
    return render_template('index.html')


@app.route('/recommend_stocks', methods=['GET'])
def recommend_stocks():
    """
    Recommends stocks
    """
    print('..enterprise has engaged..')
    pre_selected_stocks = analyze()
    print(pre_selected_stocks)
    optimized_stocks = optimize(pre_selected_stocks)
    return json.dumps(optimized_stocks)


@app.errorhandler(500)
def server_error(e):
    """
    Log the error and stacktrace.
    """
    logging.exception(f'An error occurred during a request: {e}')
    return 'An internal error occurred.', 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
