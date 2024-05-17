import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

cols = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "class"]
df = pd.read_csv("seeds_dataset.txt", names=cols, sep="\s+")
# see fcc-seeds-initial_plots.py for initial plot data setup and examination (3:39:20)

X = df[cols[:-1]].values
kmeans = KMeans(n_clusters = 3).fit(X)

# PCA - Principal Component Analysis
pca = PCA(n_components=2)
transformed_x = pca.fit_transform(X)
# plt.scatter(transformed_x[:, 0], transformed_x[:, 1])         # plot PCA transformed dataset
# K Means classes
kmeans_pca_df = pd.DataFrame(np.hstack((transformed_x, kmeans.labels_.reshape(-1, 1))), columns=["pca1", "pca2", "class"])
# Truth classes
truth_pca_df = pd.DataFrame(np.hstack((transformed_x, df["class"].values.reshape(-1, 1))), columns=["pca1", "pca2", "class"])
# scatter plots of transformed dataset ('truth') and kmeans on PCA dataset
# sns.scatterplot(x="pca1", y="pca2", hue='class', data=kmeans_pca_df)
sns.scatterplot(x="pca1", y="pca2", hue='class', data=truth_pca_df)

plt.show()
