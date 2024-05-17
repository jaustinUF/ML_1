# First four models: KNN, Naive Bayes, Logistic Regression, Support Vector Machine
# through 1:39:40
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# dataset source: https://archive.ics.uci.edu/dataset/159/magic+gamma+telescope
cols = ["fLength", "fWidth", "fSize", "fConc", "fConc1", "fAsym", "fM3Long", "fM3Trans", "fAlpha", "fDist", "class"]
df = pd.read_csv('magic04.data', names = cols)
# print(df.info()) # show schema
# in 'class' column: change 'g' and 'h' to numbers so working with array (?)
df["class"] = (df["class"] == "g").astype(int) # map 'class': 'g' > 1, else ('h') > 0

# 'shuffle' entire dataset randomly and divide into three parts
train, valid, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))]) # train gets 60%, valid 20%, test 20%

# Preprocessing: https://scikit-learn.org/stable/modules/preprocessing.html
# (36:12) need for normalization
def scale_dataset(dfl, oversample=False):
    X = dfl[dfl.columns[:-1]].values # array of arrays of row values in first 10 columns
    y = dfl[dfl.columns[-1]].values  # array of values in last column ('class')
    # print(f'type x: {type(x)}\n{x}\n\ntype y: {type(y)}\n{y}')

    # StandardScaler from sklearn.preprocessing: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
    scaler = StandardScaler()       # scale dataset around mean/sd for each column
    X = scaler.fit_transform(X)     # StandardScaler method: fit and transform data

    # Imbalanced data sets: https://imbalanced-learn.org/stable/introduction.html
    # Over-sampling: https://imbalanced-learn.org/stable/over_sampling.html
    if oversample:
        ros = RandomOverSampler()
        X, y = ros.fit_resample(X, y)
        # print(f'y: {y}\n')
    data = np.hstack((X, np.reshape(y, (-1, 1))))
    return data, X, y
    # print(f'data: {data}\nX: {X}\ny: {y}')

train, X_train, y_train = scale_dataset(train, oversample=True)
valid, X_valid, y_valid = scale_dataset(valid, oversample=False)
test, X_test, y_test = scale_dataset(test, oversample=False)
# datasets are properly formatted (44:23

# Model: KNN, K-Nearset Neighbors (K = number of neighbors checked)
# find straight-line (Euclidean) distance from (K) nearest datapoints.
knn_model = KNeighborsClassifier(n_neighbors=1) # instantiate class
knn_model.fit(X_train, y_train)                 # fit (train) model
# prediction
y_pred = knn_model.predict(X_test)              # get predictions from trained model
# Precision and recall: https://en.wikipedia.org/wiki/Precision_and_recall
print("KNN prediction:")
print(classification_report(y_test, y_pred))    # see wikipedia

#  Naive Bayes - !:08:43 in video
nb_model = GaussianNB()                         # instantiate class
nb_model = nb_model.fit(X_train, y_train)       # fit (train) model
y_pred = nb_model.predict(X_test)               # get predictions from trained model
# Note: 'predict' method in naive_bayes.py; don't know why PyCharm is complaining.
print("naive Bayes prediction:")
print(classification_report(y_test, y_pred))    # see comments in KNN above

# Logistic Regression (1:19:26)
# ChatGPT: https://chat.openai.com/c/5c7ab0d6-5ed5-42cb-93b2-561688bf9f78
lg_model = LogisticRegression()                 # same pattern as KNN above
lg_model = lg_model.fit(X_train,y_train)
y_pred = lg_model.predict(X_test)
print("Logistic Regression prediction:")
print(classification_report(y_test, y_pred))

# Support Vector Machine (1:29:15)
# Video: https://www.youtube.com/watch?v=_YPScrckx28
svm_model = SVC()                               # same pattern as KNN above
svm_model = svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
print("SVC prediction:")
print(classification_report(y_test, y_pred))    # best accuracy of these four models

# Neural Networks (1:39:44):
#   grid version in fcc_MAGIC_example_NN.py)
#   single-arg version in fcc_MAGIC_NN_one_run.py



