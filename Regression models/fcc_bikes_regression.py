# Neural net regression in fcc_bikes_nn_regression.py
#  encoding='windows-1252'
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.linear_model import LinearRegression

def show_scatter_plots():
    # plot columns against 'bike_count'
    for label in df.columns[1:]:  # all the features (label) against 'bike_count'
        plt.scatter(df[label], df['bike_count'])  # uses matplotlib.pyplot scatter plot (x, y)
        plt.title(label)
        plt.ylabel("Bike Count at Noon")
        plt.xlabel(label)
        plt.show()

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
# Single linear regression
_, x_train_temp, y_train_temp = get_xy(train, 'bike_count', x_labels=['temp'])
_, x_val_temp, y_val_temp = get_xy(val, 'bike_count', x_labels=['temp'])
_, x_test_temp, y_test_temp = get_xy(test, 'bike_count', x_labels=['temp'])
# three data sets available!
# Single linear regression
'''
temp_reg = LinearRegression()                               # instantiate model
temp_reg.fit(x_train_temp, y_train_temp)                    # train model
print(f'x coefficient: {temp_reg.coef_}, y intercept: {temp_reg.intercept_}')
print(temp_reg.score(x_test_temp, y_test_temp))             # coefficient of determination
# coefficient of determination is comparison of residual with average: is regression better than average?
## plot regression line on scatter plot
plt.scatter(x_train_temp, y_train_temp, label="Data", color="blue")
x = tf.linspace(-20, 40, 100)                               # x-axis scale, and values for regression
# x is 100 values from -20 to 40
plt.plot(x, temp_reg.predict(np.array(x).reshape(-1, 1)), label="Fit", color="red", linewidth=3)
plt.legend()
plt.title("Bikes vs Temp")
plt.ylabel("Number of bikes")
plt.xlabel("Temp")
plt.show()
'''

# Multiple Linear Regression
_, x_train_all, y_train_all = get_xy(train, 'bike_count', x_labels=df.columns[1:])
_, x_val_all, y_val_all = get_xy(val, 'bike_count', x_labels=df.columns[1:])
_, x_test_all, y_test_all = get_xy(test, 'bike_count', x_labels=df.columns[1:])
all_reg = LinearRegression()
all_reg.fit(x_train_all, y_train_all)
print(all_reg.score(x_test_all, y_test_all))



