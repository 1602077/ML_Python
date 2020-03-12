""" HIERARCHIAL CLUSTERING - AGGLOMERATIVE (RANDOM DATASET)"""
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.cluster import hierarchy
from scipy.spatial import distance_matrix
from matplotlib import pyplot as plt
from sklearn import manifold, datasets
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets.samples_generator import make_blobs

#GENERATING DATA
X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2,-1], [1,1], [10,4]], cluster_std=0.9)
#plt.scatter(X1[:,0], X1[:,1], marker='o')
agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')
"""
Agglomerative clustering requires two inputs:
n_clusters: # of clusters to generate
linkage: defines distance between sets of observation
"""
agglom.fit(X1,y1)
plt.figure(figsize=(6,4)) #inches
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0) #scaling of data to reduce scattering
X1 = (X1 - x_min)/(x_max - x_min)
for i in range(X1.shape[0]):
    plt.text(X1[i,0], X1[i,1], str(y1[i]), color = plt.cm.nipy_spectral(agglom.labels_[i] / 10.), fontdict={'weight': 'bold', 'size': 9})
    #replaces data points with respective cluster value & each cluster is  different colour.
plt.xticks([])
plt.yticks([])
plt.scatter(X1[:,0], X1[:,1], marker='.')
#plt.show()

#Dendogram for Agglo. Hier. Clust - distance matrix constains dist. from each point to eacother in dataset
dist_matrix = distance_matrix(X1,X1)
print(dist_matrix) #Check for symmetry, diag=0
Z = hierarchy.linkage(dist_matrix, 'complete') #using linkage class

#CREATING DENDROGRAM TO VISUALISE CLUSTERING
dendro = hierarchy.dendrogram(Z)
plt.savefig("AggloClust_nodata_dendro.pdf")

