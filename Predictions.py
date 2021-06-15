# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 00:55:41 2021

@author: User
"""

import pickle

#Loading the model
with open('model.pkl', 'rb') as f:
    model, cv, tfidf = pickle.load(f)

import Clean

#Method to clean the input and return the prediction
def predictions(df):
    df['cleaned_review'] = df['review'].apply(Clean.clean_text)
    cleaned_reviews = df['cleaned_review']
    cleaned_reviews = cv.fit_transform(cleaned_reviews)
    cleaned_reviews = tfidf.fit_transform(cleaned_reviews)
    cleaned_reviews.sort_indices()
    pred = model.predict(cleaned_reviews)
    return pred

pos_neg = {0 : 'negative' , 1 : 'positive'}

#Method to print whether the predictions was +ive or -ive
def review_classification(y_pred):
    print(y_pred)
    y_pred[ y_pred >= 0.5 ]  =  1
    review_class = y_pred.astype('int')
    review_class = [ pos_neg[p[0]] for p in review_class ]
    print(review_class)
