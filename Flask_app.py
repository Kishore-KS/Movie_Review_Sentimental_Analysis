# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:07:21 2021

@author: User
"""

from flask import Flask, request, render_template 
app = Flask(__name__)

import Predictions

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods =["GET", "POST"])
def predict():
    review = request.form.get("review")
    pred = Predictions.predictions(review)
    review_class = Predictions.review_classification(pred)
    output = review_class[1]
    return render_template("index.html",pred=output)
  
if __name__=='__main__':
    app.debug = True
    app.run()
