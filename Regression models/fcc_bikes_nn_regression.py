#  encoding='windows-1252'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
from sklearn.preprocessing import StandardScaler
import copy
import seaborn as sns
from sklearn.linear_model import LinearRegression
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf

def get_xy(dff, y_label, x_labels=None):
    '''Get x amd y datasets'''
    dff = copy.deepcopy(dff)
    if x_labels is None:
        x = dff[[c for c in dff.columns if c != y_label]].values    # all columns except this one
    else:
        if len(x_labels) == 1:
            x = dff[x_labels[0]].values.reshape(-1, 1)
        else:
            x = dff[x_labels].values
    y = dff[y_label].values.reshape(-1, 1)
    data = np.hstack((x, y))
    return data, x, y

# get data, drop and rename columns
df = pd.read_csv("SeoulBikeData.csv", encoding='windows-1252').drop(["Date", "Holiday", "Seasons"], axis=1)
df.columns = ["bike_count", "hour", "temp", "humidity", "wind", "visibility", "dew_pt_temp", "radiation", "rain", "snow", "functional"]
df['functional'] = (df["functional"] == 'Yes').astype(int)  # change to integer (yes = 1 else = 0)
df = df[df["hour"] == 12]           # keep only noon-time rows; reduces row count from 8760 to 365
df = df.drop(['hour'], axis='columns')                      # drop 'hour' column
# show_scatter_plots()
# (2:42:58) look at plots: see relationship to 'bike_count'?
#   "wind": low coefficient of determination: r2 = 1 - TSS/RSS  (2:28:40)
#   "visibility": doesn't look linear
#   "functional": no relationship, straight vertical line
df = df.drop(["wind", "visibility", "functional"], axis=1)
# shuffle and split df into datasets: training (60%), validation (20%), test (20%)
train, val, test = np.split(df.sample(frac=1), [int(0.6*len(df)), int(0.8*len(df))])
# Build temperature data sets
_, x_train_temp, y_train_temp = get_xy(train, 'bike_count', x_labels=['temp'])
_, x_val_temp, y_val_temp = get_xy(val, 'bike_count', x_labels=['temp'])
_, x_test_temp, y_test_temp = get_xy(test, 'bike_count', x_labels=['temp'])
# Build all-data data sets
_, x_train_all, y_train_all = get_xy(train, 'bike_count', x_labels=df.columns[1:])
_, x_val_all, y_val_all = get_xy(val, 'bike_count', x_labels=df.columns[1:])
_, x_test_all, y_test_all = get_xy(test, 'bike_count', x_labels=df.columns[1:])

# Neural Net Regression - single layer (2:54:49)
def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['val_loss'], label='val_loss')
  plt.xlabel('Epoch')
  plt.ylabel('MSE')
  plt.legend()
  plt.grid(True)
  plt.show()
# can use layers to do different things ... like normalize
temp_normalize = tf.keras.layers.Normalization(input_shape=(1,), axis=None)
temp_normalize.adapt(x_train_temp.reshape(-1))
# instantiate (one dense-layer model" ('neuron')
temp_nn_model = tf.keras.Sequential([
    temp_normalize,
    tf.keras.layers.Dense(1)
])
# 'compile' to specify a loss, metrics (a list), and an optimizer.
temp_nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')
#  (2:57:44) Lin Regression using a Neuron
# train the model (fit it to the data)
'''
history = temp_nn_model.fit(x_train_temp.reshape(-1), y_train_temp, verbose = 0,
                epochs = 1000, validation_data=(x_val_temp, y_val_temp))
plot_loss(history)                                # uncomment to see plot
'''
# run previous linear regression using nn model
'''
plt.scatter(x_train_temp, y_train_temp, label="Data", color="blue")
x = tf.linspace(-20, 40, 100)                               # x-axis scale, and values for regression
# x is 100 values from -20 to 40
plt.plot(x, temp_nn_model.predict(np.array(x).reshape(-1, 1)), label="Fit", color="red", linewidth=3)
plt.legend()
plt.title("Bikes vs Temp")
plt.ylabel("Number of bikes")
plt.xlabel("Temp")
plt.show()
'''
# Neural Net (3:00:15)
# Regression NN using Tensorflow(three layers using multiple nodes (32)
# fit temperature data
'''nn_model = tf.keras.Sequential([
    temp_normalize,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
'''
# fit model to temperature data set
'''history = nn_model.fit(
    x_train_temp, y_train_temp,
    validation_data=(x_val_temp, y_val_temp),
    verbose=0, epochs=100
)
plot_loss(history)'''
# Neural net model shows that temperature regression is not linear ... which one can
#   see from looking at scatter plot. Violates several of linear regression assumptions (homoscedasticity)
# Use all data (3:04:21)

all_normalize = tf.keras.layers.Normalization(input_shape=(6,), axis=-1)
all_normalize.adapt(x_train_all)

nn_model = tf.keras.Sequential([
    all_normalize,
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])
# 'Adam optimization is a stochastic gradient descent method ..."
nn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mean_squared_error')
# fit model to temperature data set
history = nn_model.fit(
    x_train_all, y_train_all,
    validation_data=(x_val_all, y_val_all),
    verbose=0, epochs=100
)
# (3:06:41) tutorial reaches back to get lin reg MSE, to compare with nn MSE below
y_pred_nn = nn_model.predict(x_test_all)
print('Mean square error: ', np.square(y_pred_nn-y_test_all).mean())
# plot_loss(history) # (3:05:22)
# (3:09:00) scatter plot comparing lin reg with nn
ax = plt.axes(aspect="equal")
# plt.scatter(y_test_all, y_pred_lr, label="Lin Reg Preds")
plt.scatter(y_test_all, y_pred_nn, label="NN Preds")
plt.xlabel("True Values")
plt.ylabel("Predictions")
lims = [0, 1800]
plt.xlim(lims)
plt.ylim(lims)
plt.legend()
_ = plt.plot(lims, lims, c="red")
plt.show()
# (3:12:19)

