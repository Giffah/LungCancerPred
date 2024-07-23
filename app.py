import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route('/details' , methods=['GET']) 
def details():
    return render_template('Details.html')

@flask_app.route("/predict", methods = ["POST"])
def predict():
    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    prediction = model.predict(features)[0]

    # Map the prediction to risk levels
    risk_level = {2: "high risk", 1: "medium risk", 0: "low risk"}
    prediction_text = risk_level.get(prediction, "unknown risk level")

    return render_template("index.html", prediction_text=f"You have a {prediction_text} of getting lung cancer")


if __name__ == "__main__":
    flask_app.run(debug=True)