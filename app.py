"""
Flask application that serves as the API
"""
#!/usr/bin/env python
import logging
import json

from flask import Flask
from flask import render_template
from core.analyzer import run as analyze
from core.optimizer import run as optimize

# import argparse

app = Flask(__name__, static_url_path="", static_folder="static")


@app.route("/")
def template():
    """
    Returns base template
    """
    return render_template("index.html")


@app.route("/recommend_stocks/", methods=["GET"])
def recommend_stocks():
    """
    Recommends stocks
    """
    pre_selected_stocks = analyze()
    print(pre_selected_stocks)
    optimized_stocks = optimize(pre_selected_stocks, value=400)
    return json.dumps(optimized_stocks)


@app.errorhandler(500)
def server_error(error):
    """
    Log the error and stacktrace.
    """
    logging.exception("An error occurred during a request: %s", error)
    return "An internal error occurred.", 500


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--budget", type=int)
    # args = parser.parse_args()
    # app.config["budget"] = args.budget
    app.run(host="0.0.0.0", port=5000, debug=True)
