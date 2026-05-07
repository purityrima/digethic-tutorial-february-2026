from flask import (
    Flask,
    request,
)  # We need "request" because "request.args.get()" comes from Flask’s request object.
from flask_cors import CORS

import pandas as pd
import pickle

with open(
    "data/models/baumethoden_lr.pickle", "rb"
) as file:  # rb = read binary, because pickle files are binary files, not text files.
    trained_model = pickle.load(file)  # pickle.load is used

# Task 2

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


# Task 3
@app.route("/predict", methods=["GET"])
def predict():
    zylinder = float(request.args.get("zylinder"))
    ps = float(request.args.get("ps"))
    gewicht = float(request.args.get("gewicht"))
    beschleunigung = float(request.args.get("beschleunigung"))
    baujahr = float(request.args.get("baujahr"))

    print(
        f"Received request with zylinder: {zylinder}, ps: {ps}, gewicht: {gewicht}, beschleunigung: {beschleunigung}, baujahr: {baujahr}"
    )

    prediction_data = [[zylinder, ps, gewicht, beschleunigung, baujahr]]
    # prediction_data = [[ps, gewicht]]
    prediction = trained_model.predict(prediction_data)

    return {"result": float(prediction[0])}
