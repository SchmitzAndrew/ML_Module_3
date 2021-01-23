import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix

#Results
from Results import displayData

#Naive Bayes:
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report

#LVSM
from sklearn.linear_model import SGDClassifier

#Logistic Regression
from sklearn.linear_model import LogisticRegression

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

#allows access to results
global accuracy_NB
global accuracy_LSVM
global accuracy_LogR
global accuracy_DT
global header_list
def load_data():
    header_list = ['Post', 'Category', 'Date', 'Last_Reply', 'Replies', 'Username']
    data = pd.read_csv('data.csv', names = header_list)
    print(data.columns)
    print(data.head())
    return data.copy()

header_list = ['Post', 'Category', 'Date', 'Last_Reply', 'Replies', 'Username']
df = pd.read_csv('data.csv', names = header_list)

def model_naive_bayes(df):
    test_df = df
    X = test_df.Post
    Y = test_df.Category
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=11)
    naive_bayes_categories = ['Post', 'Category', 'Result']

    #naive-bayes
    nb = Pipeline([('vect', CountVectorizer()),
                   ('tfidf', TfidfTransformer()),
                   ('clf', MultinomialNB()),
                   ])
    nb.fit(X_train, Y_train)


    y_pred = nb.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, Y_test))
    res1311 = accuracy_score(y_pred, Y_test)
    print(classification_report(Y_test, y_pred, target_names = naive_bayes_categories))
    return res1311

def LSVM(df):
    test_df2 = df
    X = test_df2.Post
    Y = test_df2.Category
    X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=11)
    LSVM_categories = ['Post', 'Category', 'Result1']
    sgd = Pipeline([('vect', CountVectorizer()),
                    ('tfidf', TfidfTransformer()),
                    ('clf',
                     SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=5, tol=None)),
                    ])
    sgd.fit(X_train, Y_train)

    y_pred = sgd.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    res1321 = accuracy_score(y_pred, y_test)
    print(classification_report(y_test, y_pred, target_names= LSVM_categories))
    return res1321

def Log_Regression(df):
    test_df3 = df
    X = test_df3.Post
    Y = test_df3.Category
    X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=11)
    logR_categories = ['Post', 'Category', 'Result1']
    logreg = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', LogisticRegression(n_jobs=1, C=1e5)),
                       ])
    logreg.fit(X_train, Y_train)

    y_pred = logreg.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    res1331 = accuracy_score(y_pred, y_test)
    print(classification_report(y_test, y_pred, target_names=logR_categories))
    return res1331

def Decision_Tree(df):
    test_df4 = df
    X = test_df4.Post
    Y = test_df4.Category
    X_train, X_test, Y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=11)
    CV_categories = ['Post', 'Category', 'Result1']
    dtree = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', DecisionTreeClassifier(random_state=0)),
                      ])
    dtree.fit(X_train, Y_train)

    y_pred = dtree.predict(X_test)

    print('accuracy %s' % accuracy_score(y_pred, y_test))
    res1341 = accuracy_score(y_pred, y_test)
    print(classification_report(y_test, y_pred, target_names=CV_categories))
    return res1341

accuracy_NB = model_naive_bayes(df)
accuracy_LSVM = LSVM(df)
accuracy_LogR = Log_Regression(df)
accuracy_DT = Decision_Tree(df)
displayData(accuracy_NB, accuracy_LSVM, accuracy_LogR, accuracy_DT)
