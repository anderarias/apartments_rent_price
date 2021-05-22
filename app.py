from flask import Flask,request, url_for, redirect, render_template, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

model = joblib.load("forest_reg_model.pkl")
scaler = joblib.load("scaler.pkl")
hot_encoder = joblib.load("hot_encoder.pkl")

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
    features = [x for x in request.form.values()]
    
    distance = int(features[0])
    years = int(features[1])
    area = int(features[2])
    madori = features[3]

    x_test_num_raw = np.array([distance, years, area]).reshape(1,-1)
    x_test_num_scaled = scaler.transform(x_test_num_raw)
    x_test_cat_raw = np.array([madori]).reshape(1,-1)
    x_test_cat_enc = hot_encoder.transform(x_test_cat_raw).toarray()
    x_test = np.concatenate((x_test_num_scaled, x_test_cat_enc), axis=1)

    prediction = model.predict(x_test).item()
    
    return render_template('index.html',pred='{:.2f}'.format(prediction))


if __name__ == '__main__':
    app.run(debug=True)