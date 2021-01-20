import pandas as pd
import logging
import numpy as np
from numpy import random
import gensim
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import re

#Naive Bayes:
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer


def load_data():
    header_list = ['Post', 'Category', 'Date', 'Last_Reply', 'Replies', 'Username']
    data = pd.read_csv('data.csv', names = header_list)
    print(data.columns)
    print(data.head())
    df = data.copy()
    model(df)

def model(df):
    test_df = df
    X = test_df.Post
    Y = test_df.Category
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    my_categories = ['Post', 'Category']
    print(X_train.head())
    print(Y_train.head())

    #naive-bayes
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])
    nb.fit(X_train, Y_train)

    from sklearn.metrics import classification_report
    y_pred = nb.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, Y_test))
    res1311 = accuracy_score(y_pred, Y_test)
    print(classification_report(Y_test, y_pred, target_names = my_categories))


