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

def load_data():
    header_list = ['Post', 'Category', 'Date', 'Last_Reply', 'Replies', 'Username']
    data = pd.read_csv('data.csv', names = header_list)
    print(data.columns)
    print(data.head())
    df = data.copy()








# test_df = data.copy()
#     X = test_df.post
#     y = test_df.Category

