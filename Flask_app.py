# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:07:21 2021

@author: User
"""

from flask import Flask, request, render_template 
app = Flask(__name__)

import Predictions

@app.route('/', methods =["GET", "POST"])
def gfg():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       review = request.form.get("review")
       pred = Predictions.predictions(review)
       review_class = Predictions.review_classification(pred)
       return "The review has a "+review_class[0]+" probability of being "+review_class[1]
    return render_template("index.html")
  
if __name__=='__main__':
    app.debug = True
    app.run()
