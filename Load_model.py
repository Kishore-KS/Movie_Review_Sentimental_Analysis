# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 21:07:21 2021

@author: User
"""
import pandas as pd
import methods

'''string_input = input("Enter your review: ")
string_df = pd.DataFrame([string_input],columns=["review"])
y_pred = sam.predictions(string_df)
sam.review_classification(y_pred)'''

from flask import Flask, request, render_template 
  
# Flask constructor
app = Flask(__name__)   
  
# A decorator used to tell the application
# which URL is associated function
@app.route('/', methods =["GET", "POST"])
def gfg():
    if request.method == "POST":
       # getting input with name = fname in HTML form
       review = request.form.get("review")
       string_df = pd.DataFrame([review],columns=["review"])
       y_pred = methods.predictions(string_df)
       review_class = methods.review_classification(y_pred)
       return "Your review classification ",y_pred," ",review_class
    return render_template("index.html")
  
if __name__=='__main__':
    app.debug = True
    app.run()
