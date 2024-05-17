# through preprocessing to datasets properly formatted.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler

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
'''dss = df.sample(frac=.0003)
print(scale_dataset(dss, True))
print(f'\n-   -   -   -   -   -   \n')
print(scale_dataset(dss, False))'''

''' # look at data to find differences between gamma and hadron samples
for label in cols[:4]:
    plt.hist(df[df['class'] == 1][label], color='blue', label='gamma', alpha=0.7, density=True)
    plt.hist(df[df['class'] == 0][label], color='red', label='hadron', alpha=0.7, density=True)
    plt.title(label)
    plt.ylabel("Probability")
    plt.xlabel(label)
    plt.legend()
    plt.show()'''
# print(df['class'].unique())
# print(df.head())


