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