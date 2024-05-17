import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

cols = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "class"]
df = pd.read_csv("seeds_dataset.txt", names=cols, sep="\s+")
# print(df.head())
# see fcc-seeds-initial_plots.py for initial plot data setup and examination (3:39:20)

x = "compactness"
# x = "perimeter"
y = "asymmetry"
X = df[[x, y]].values

kmeans = KMeans(n_clusters=3).fit(X)
# clusters = kmeans.labels_
# print(clusters)                                 # display the cluster valies for each x,y
# print(df['class'].values)                       # compare with the class values
# print(clusters.reshape(-1,1))                   # reshape cluster into array of 1D arrays
# print(np.hstack((X, clusters.reshape(-1, 1))))  # join x and y values with cluster label

# build dataframe from kmeans clusters
cluster_df = pd.DataFrame(np.hstack((X, kmeans.labels_.reshape(-1,1))), columns=[x, y, 'class'])
# # K Means class
sns.scatterplot(x=x, y=y, hue='class', data=cluster_df)
# Original class
# sns.scatterplot(x=x, y=y, hue='class', data=df)

# Higher Dimensions
# X = df[cols[:-1]].values
# kmeans = KMeans(n_clusters = 3).fit(X)
# cluster_df = pd.DataFrame(np.hstack((X, kmeans.labels_.reshape(-1, 1))), columns=df.columns)
# sns.scatterplot(x=x, y=y, hue='class', data=cluster_df)




plt.show()






# bt = time.time()
# print('Run time:', time.strftime("%H:%M:%S", time.gmtime(time.time()-bt)))

