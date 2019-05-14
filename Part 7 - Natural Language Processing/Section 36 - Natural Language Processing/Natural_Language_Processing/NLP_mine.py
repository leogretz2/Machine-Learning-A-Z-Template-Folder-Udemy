# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts
import re
import nltk
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1000):
    review = re.sub('[^a-zA-Z]', ' ',  dataset['Review'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
    
# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Results
results = []
name = ''
for i in range(0, 8):
    if i == 0:
        from sklearn.linear_model import LogisticRegression
        classifier = LogisticRegression(random_state = 0)
        name = 'LR:'
    if i == 1:
        from sklearn.neighbors import KNeighborsClassifier
        classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
        name = 'KNN:'
    if i == 2:
        from sklearn.svm import SVC
        classifier = SVC(kernel = 'linear', random_state = 0)
        name = 'SVM:'
    if i == 3:
        from sklearn.naive_bayes import GaussianNB
        classifier = GaussianNB()
        name = 'Naive Bayes:'
    if i == 4:
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
        name = 'Decision Tree:'
    if i == 5:
        from sklearn.ensemble import RandomForestClassifier
        classifier = RandomForestClassifier(n_estimators = 100, criterion = 'entropy', random_state = 0)
        name = 'Random Forest:'
    if i == 6:
        from sklearn.tree import DecisionTreeClassifier
        classifier = DecisionTreeClassifier()
        name = 'CART' 
#    if i == 7:
#        from nltk import maxent
#        classifier = maxent.MaxentClassifier.train(X_train, algorithm=None, trace=3, encoding=None, labels=None, gaussian_prior_sigma=0, max_iter = 3)
#        name = 'Maximum Entropy'
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred)
    TN = cm[0][0]
    FN = cm[1][0]
    TP = cm[1][1]
    FP = cm[0][1]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * precision * recall / (precision + recall)
    data = [name, accuracy, precision, recall, F1]
    results.append(data)





