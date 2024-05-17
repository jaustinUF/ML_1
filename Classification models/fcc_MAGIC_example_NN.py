# Neural Networks
#   the grid version (multi-arguments) started at 1:56:50 (version in co-lab)
#  (first four models are in fcc_MAGIC_example_2.py)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import time
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
def plot_history(history):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(history.history['loss'], label='loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Binary crossentropy')
    ax1.grid(True)

    ax2.plot(history.history['accuracy'], label='accuracy')
    ax2.plot(history.history['val_accuracy'], label='val_accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True)

    plt.show()

def train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epoch_cnt):
    nn_model = tf.keras.Sequential([
        # tf.keras.layers.Dense(num_nodes, activation='relu', input_shape=(10,)),  # 'input_shape' sem to cause error?
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(num_nodes, activation='relu'),
        tf.keras.layers.Dropout(dropout_prob),
        tf.keras.layers.Dense(1, activation='sigmoid')
        ])
    nn_model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss='binary_crossentropy',
                     metrics=['accuracy'])
    # train and save training 'history'
    history = nn_model.fit(
        X_train, y_train, epochs=epoch_cnt, batch_size=batch_size, validation_split=0.2, verbose=0)
    return nn_model, history

least_val_loss = float('inf')
least_loss_model = None
epochs = 100
batch = 1
print('Start:', time.strftime("%H:%M:%S", time.localtime()))
bt = time.time()
for num_nodes in [16, 32, 64]:
    for dropout_prob in [0, 0.2]:
        for lr in [0.01, 0.005, 0.001]:
            for batch_size in [32, 64, 128]:
                print(f"batch {batch}: {num_nodes} nodes, dropout {dropout_prob}, lr {lr}, batch size {batch_size}")
                batch += 1
                model, history = train_model(X_train, y_train, num_nodes, dropout_prob, lr, batch_size, epochs)
                # plot_history(history)
                val_loss = model.evaluate(X_valid, y_valid)[0]
                if val_loss < least_val_loss:
                    least_val_loss = val_loss
                    least_loss_model = model
print('End:', time.strftime("%H:%M:%S", time.localtime()))
print('Run time:', time.strftime("%H:%M:%S", time.gmtime(time.time()-bt)))

y_pred = least_loss_model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int).reshape(-1,)
print(classification_report(y_test, y_pred))


