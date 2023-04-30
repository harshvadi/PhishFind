from flask import Flask, render_template, request
import joblib
import pandas as pd
from prediction.features import exportToDataSet

app = Flask("fake_website_identifier")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/urlPage")
def urlPage():
    return render_template("urlPage.html")


@app.route("/urlPage", methods=["POST"])
def submit():
    url = request.form["url"]
    exportToDataSet(url)

    knn = joblib.load("./prediction/knn.pkl")
    prediction_knn = knn.predict(pd.read_csv("test.csv"))

    svm = joblib.load("./prediction/svm.pkl")
    prediction_svm = svm.predict(pd.read_csv("test.csv"))

    forest = joblib.load("./prediction/forest.pkl")
    prediction_forest = forest.predict(pd.read_csv("test.csv"))

    tree = joblib.load("./prediction/tree.pkl")
    prediction_tree = tree.predict(pd.read_csv("test.csv"))

    prediction = (
        prediction_knn[0]
        + prediction_svm[0]
        + prediction_forest[0]
        + prediction_tree[0]
    )

    print(prediction)

    # majority voting
    prediction = prediction / 2

    return render_template("urlPage.html", prediction=prediction, url=url)


app.run("localhost", "5000", debug=True)
