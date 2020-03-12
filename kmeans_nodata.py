""" K-Means - Customer Segmentation (Classification) Unsupervised learning using unsupervised data """
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs
import matplotlib.pyplot as plt
from IPython.display import display

#Create own dataset - use np.randmom.seed where the seed is set to 0
np.random.seed(0)
#Use make_blobs to create random clusters of points, cluster_std is the standard deviation of the clusters
#make_blobs outputs two array one of shape [n_samples, n_features] and another [n_samples]

X, y = make_blobs(n_samples=5000, centers=[[4,4], [-2,-1], [2,-3], [1,1]], cluster_std=0.9)
plt.scatter(X[:,0], X[:,1], marker='.') #Visualising data
#plt.show()

#SETTING UP K-MEANS
k_means = KMeans(init="k-means++", n_clusters=4, n_init=12)
#k-means++ is the type of initilation of centroids
#n_init is the # of iterations
k_means.fit(X)
k_means_labels = k_means.labels_
k_means_cluster_centers = k_means.cluster_centers_ #Gets labels of each point and coords of cluster centers

#VISUAL PLOT
fig = plt.figure(figsize=(6, 4))
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(k_means_labels))))
ax = fig.add_subplot(1, 1, 1)

#Loop k from 0 to 3 - # of clusters
for k, col in zip(range(len([[4,4], [-2, -1], [2, -3], [1, 1]])), colors):
    my_members = (k_means_labels == k )
    cluster_center = k_means_cluster_centers[k] #Define cluster center
    #plots data points w/ color col.
    ax.plot(X[my_members, 0], X[my_members, 1], 'w', markerfacecolor=col, marker='.')
    #plots centroids with specified col in darker outline
    ax.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col, markeredgecolor='k', markersize=6)
ax.set_title('KMeans')
ax.set_xticks(()) #no ticks
ax.set_yticks(())
plt.show()

