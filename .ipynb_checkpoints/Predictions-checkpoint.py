# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 00:55:41 2021

@author: User
"""

import pickle
from keras.models import load_model

#Loading the model
model = load_model('models/movie_review_prediction.h5')
tfidf = pickle.load(open("models/tfidf.pkl", "rb"))
cv = pickle.load(open("models/cv.pkl", "rb"))

import Clean

import pandas as pd
#Method to clean the input and return the prediction
def predictions(review):
    df = pd.DataFrame([review],columns=["review"])
    df['cleaned_review'] = df['review'].apply(Clean.clean_text)
    cleaned_reviews = df['cleaned_review']
    cleaned_reviews = cv.transform(cleaned_reviews)
    cleaned_reviews = tfidf.transform(cleaned_reviews)
    cleaned_reviews.sort_indices()
    pred = model.predict(cleaned_reviews)
    return pred

pos_neg = {0 : 'negative' , 1 : 'positive'}

#Method to print whether the predictions was +ive or -ive
def review_classification(y_pred):
    pred_p = str(y_pred)
    y_pred[ y_pred >= 0.5 ]  =  1
    review_class = y_pred.astype('int')
    review_class = [ pos_neg[p[0]] for p in review_class ]
    return pred_p, str(review_class[0])
