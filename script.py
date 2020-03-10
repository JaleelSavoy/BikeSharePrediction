import numpy as np
import keras
import pickle 
import os
import flask
from flask import Flask, render_template, request

app=Flask(__name__)

@app.route('/')
def home():
    return flask.render_template('input.html')

@app.route('/input')
def index():
    return flask.render_template('input.html')


def ValuePredictor(prediction_input):
    data = np.array(prediction_input).reshape(1,13)
    nnet = pickle.load(open("bike_pred_model.sav", 'rb'))
    pred = round((nnet.predict(data))[0][0],2)
    return pred

@app.route('/output',methods = ['POST'])
def result():
    if request.method == 'POST':
        data_dict = request.form.to_dict()
        data_dict['windspeed'] = float(data_dict['windspeed'])/67
        data_dict['windspeed_yest'] = float(data_dict['windspeed_yest'])/67
        data_dict['temp'] = (float(data_dict['temp'])+8)/(39+8)
        data_dict['temp_yest'] = (float(data_dict['temp_yest'])+8)/(39+8)
        data = list(data_dict.values())
        data = list(map(float, data))
        result = ValuePredictor(data)
        return render_template("output.html",prediction=result)

if __name__ == "__main__":
    app.run(debug=True)