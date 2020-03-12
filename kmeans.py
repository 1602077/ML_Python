""" K-Means Clustering - Customer Segmentatino """
import numpy as np
import pandas as pd
from IPython.display import display

cust_df = pd.read_csv("Cust_Segmentation.csv")
#display(cust_df.head())
df=cust_df.drop('Address', axis=1) #dropping adress as not categorical data
display(df.head())

from sklearn.preprocessing import StandardScaler
X = df.values[:,1:]
X = np.nan_to_num(X) #normalising data
Clus_dataSet = StandardScaler().fit_transform(X)
#display(Clus_dataSet)

#MODELLING - Applying kmeans
from sklearn.cluster import KMeans
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)
k_means.fit(X)
labels = k_means.labels_
#print(labels)

#INSIGHTS
df["Clus_km"] = labels #assigning labels to each row in the df
#display(df.head(5))
display(df.groupby('Clus_km').mean()) #averages features of each cluster to check center value

#analysing distribution of customer age / income
import matplotlib.pyplot as plt
area = np.pi * (X[:, 1])*(X[:, 1])
plt.scatter(X[:,0], X[:,3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=16)
plt.ylabel('Income', fontsize=16)
plt.savefig("kmeans_scatter.pdf")

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(1, figsize=(8,6))
plt.clf() #clears current fig
ax = Axes3D(fig, rect=[0,0,.95,1], elev=48, azim=134) #azim: azimuthal viewing angleA
plt.cla() #clears axes

ax.set_xlabel('Education')
ax.set_ylabel('Age')
ax.set_zlabel('Income')
ax.scatter(X[:,1], X[:,0], X[:,3], c=labels.astype(np.float))
plt.savefig("kmeans_3D.pdf")

