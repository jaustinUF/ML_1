import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

output_dir = 'plots'
cols = ["area", "perimeter", "compactness", "length", "width", "asymmetry", "groove", "class"]
df = pd.read_csv("seeds_dataset.txt", names=cols, sep="\s+")
# print(df.head())
# initial plot code (3:39:20)
k = 1
for i in range(len(cols)-1):
  for j in range(i+1, len(cols)-1):
    x_label = cols[i]
    y_label = cols[j]
    # print(f" {k} {x_label} vs {y_label}")
    sns.scatterplot(x=x_label, y=y_label, data=df, hue='class') \
          .set(title=f"{x_label} vs {y_label} - Plot {k}")
    plt.savefig(os.path.join(output_dir, f"{k} {x_label} vs {y_label} .png"))
    plt.clf()
    k += 1


