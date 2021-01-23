import pandas as pd

def displayData(r1, r2, r3, r4):
    categories = ["Naive Bayes", "Linear SVM", "Logistic Regression", "Count Vectorizer"]
    results = pd.DataFrame({'Model': categories, 'Accuracy': [r1, r2, r3, r4]})
    results.set_index('Model')
    results.sort_values(by='Accuracy')
    print(results)


