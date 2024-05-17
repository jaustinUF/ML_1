Unsupervised learning models
PCS - Principal Component Analysis
    - used for dimensionality reduction
    - discussion 3:23:46
    - implementation 3:47:50

K-Means Clustering
    - clusters data into the number of clusters we set
    - discussion 3:13:13
    - implementation 3:40:05

K-Means and PCA Implementations 3:33:57
    - both first discussed, then implemented separately later on
    - first part of implementation is loading and labeling data

Data source: https://archive.ics.uci.edu/dataset/236/seeds
Data file: text data, seven columns of numbers
To construct the data, seven geometric parameters of wheat kernels were measured: 
1. area A, 
2. perimeter P, 
3. compactness C = 4*pi*A/P^2, 
4. length of kernel,
5. width of kernel,
6. asymmetry coefficient
7. length of kernel groove.
All of these parameters were real-valued continuous.
8. Eighth column is 'class' (1, 2, 3) ... seed (wheat kernel) type?
Note: these 'parameters' are the attributes to be used 
   