# Neural Networks (first four models are in fcc_MAGIC_example_2.py)
# NOTE: single-argument version built 1:49:51 to 1:56:47
#   produces two graphs: 'accuracy' and 'binary_crossentropy' (loss)
#   co-lab grid version in fcc_MAGIC_example_NN.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf


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

# Neural Networks (1:39:44)
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Binary crossentropy')
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

nn_model = tf.keras.Sequential([
    # tf.keras.layers.Dense(32, activation='relu', input_shape=(10,)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

nn_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='binary_crossentropy',
                 metrics=['accuracy'])
# train the model!
# 'history' saves the training data
history = nn_model.fit(
    X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=0
)
plot_loss(history)
plot_accuracy(history)
