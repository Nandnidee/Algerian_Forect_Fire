import pickle
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


application = Flask(__name__)
app = application

# import ridge and standard scaler from pickle file
ridge_model = pickle.load(open('Model/ridge.pkl', 'rb'))
scaler_model = pickle.load(open('Model/scaler.pkl', 'rb'))

@app.route("/")
def hello_world():
    return render_template('index.html')

@app.route("/prediction", methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'POST':
        temperature = float(request.form.get('temp'))
        rh = float(request.form.get('RH'))
        ws = float(request.form.get('Ws'))
        rain = float(request.form.get('Rain'))
        ffmc = float(request.form.get('FFMC'))
        dmc = float(request.form.get('DMC'))
        isi = float(request.form.get('ISI'))
        classes = float(request.form.get('Classes'))
        region = float(request.form.get('Region'))

        input_data = np.array([[temperature, rh, ws, rain, ffmc, dmc, isi]])
        input_scaled = scaler_model.transform(input_data)
        result = ridge_model.predict(input_scaled)
        return render_template('home.html', results=result[0])
    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)