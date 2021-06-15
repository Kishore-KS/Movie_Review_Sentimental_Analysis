# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 23:33:14 2021

@author: User
"""
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

sw = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(sample):
    sample = sample.lower()
    #removing all <br> tags in the review
    sample = sample.replace("<br /><br />", "")
    #removing all special characters from the review
    sample = re.sub("[^a-zA-Z]+", " ", sample)    
    sample = sample.split()
    sample = [ps.stem(s) for s in sample if s not in sw] # list comprehension
    sample = " ".join(sample)
    return sample

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
