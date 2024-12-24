from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the pre-trained model
model = joblib.load("model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Retrieve inputs from the form
    battery_power = int(request.form["battery_power"])
    ram = int(request.form["ram"])
    int_memory = int(request.form["int_memory"])
    px_width = int(request.form["px_width"])
    px_height = int(request.form["px_height"])
    mobile_wt = int(request.form["mobile_wt"])

    # Combine inputs into a single array
    features = np.array([[battery_power, ram, int_memory, px_width, px_height, mobile_wt]])

    # Predict the price range using the model
    prediction = model.predict(features)[0]  # Assumes the model outputs 0, 1, 2, or 3

    # Render the result page with the prediction
    return render_template("result.html", predicted_price=int(prediction))

if __name__ == "__main__":
    app.run(debug=True)
