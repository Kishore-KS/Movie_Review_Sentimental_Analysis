# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 23:33:14 2021

@author: User
"""

#Loading the model
from keras.models import load_model

model = load_model('movie_review_prediction.h5')

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
tfidf = TfidfTransformer()
cv = CountVectorizer(max_df = 0.5, max_features=50000)


#Method to clean the input and return the prediction
def predictions(df):
    df['cleaned_review'] = df['review'].apply(clean_text)
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
