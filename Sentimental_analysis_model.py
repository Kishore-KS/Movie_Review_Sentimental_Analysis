import numpy as np
import pandas as pd
import Clean

#Reading the training data
dataset = pd.read_csv("./Train/Train.csv")

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

y = dataset['label'].values
y = le.fit_transform(y)

dataset['cleaned_review'] =  dataset['review'].apply(Clean.clean_text)
corpus = dataset['cleaned_review'].values

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
cv = CountVectorizer(max_df = 0.5, max_features=50000)
X = cv.fit_transform(corpus)

tfidf = TfidfTransformer()
X = tfidf.fit_transform(X)

#Building the model
from keras import models
from keras.layers import Dense

#Making the network
model = models.Sequential()
model.add( Dense(16, activation="relu", input_shape = (X.shape[1],) ) )
model.add( Dense(16, activation="relu") )
model.add( Dense(1, activation="sigmoid"))

#Compiling the model
model.compile(optimizer='rmsprop', loss="binary_crossentropy", metrics=['accuracy'])

#Getting the necessary data
X_val = X[:5000]
X_train = X[5000:]
y_val = y[:5000]
y_train = y[5000:]

#Sorting indices
X_train.sort_indices()
X_val.sort_indices()

#Fitting the model
hist = model.fit(X_train, y_train, batch_size=128, epochs=2, validation_data=(X_val, y_val), shuffle=True)
print(model.evaluate(X_val, y_val))

#Dumping the model
import pickle

print("Dumping model")
pickle.dump(model, open('model.pkl','wb'))
print("Dumping cv")
pickle.dump(cv, open('cv.pkl','wb'))
print("Dumping tfidf")
pickle.dump(tfidf, open('tfidf.pkl','wb'))


