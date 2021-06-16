# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 16:01:24 2021

@author: User
"""
import Predictions

reviews = input("Enter your review: ")

pred = Predictions.predictions(reviews)
print(Predictions.review_classification(pred))