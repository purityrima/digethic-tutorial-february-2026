from flask import Flask
from flask_cors import CORS
import pandas as pd

app = Flask(__name__)  # creates the Flask application
CORS(app)  # allows cross-origin requests.


@app.route("/")  # Route 1: maps the root endpoint.
def home():
    return {"hello": "world"}  # returns JSON (file format) automatically.


@app.route("/hello_world")  # Route 2
def hello_world():
    return "<p>Hello, World!</p>"  # returns HTML text.


@app.route("/training_data")
def training_data():
    data = pd.read_csv(
        "data/auto-mpg-training-data.csv", sep=";"
    )  # reads yourthe CSV and converts it into JSON-style records:
    return data.to_dict(orient="records")  # which Flask then returns as JSON.
