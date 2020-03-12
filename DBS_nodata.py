""" Density-based Spatial Cluster of Application w/ Noise (DBSCAN) - No df"""
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

def createDataPoints(centroidLocation, numSamples, clusterDeviation):
    """
    Creates random data and stores in feature matrix X and reponse vector y.
    """
    X, y = make_blobs(n_samples=numSamples, centers=centroidLocation, cluster_std=clusterDeviation)
    X = StandardScaler().fit_transform(X)
    return X, y
X, y = createDataPoints([[4,3], [2,-1], [-1,4]], 1500, 0.5)

#MODELLING
"""
DBSCAN is based on that if a given point belongs to a cluster then there 
should be lots of points nearby that also belong to the cluster.
Uses Two Parameters: Epsilon & Minimum Points
Epsilon: specified rad. that if includes enough points if defined as a "dense area"
minimumSamples: min # of points to define as a cluster
"""
epsilon = 0.3
minimumSamples = 7
db = DBSCAN(eps=epsilon, min_samples=minimumSamples).fit(X)
labels = db.labels_
print("Labels: ", labels)

#DISTINGUISHING OUTLIERS
#Replace all elements w. True in core_samples_mask if in cluster, else False.
core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
print("Core Samples Mask: ", core_samples_mask)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
print("# of Clusters: ", n_clusters)

unique_labels = set(labels) #Remove rep. of labels by turning into a set
print("Unique Labels: ", unique_labels)

#DATA VISUALISATION
colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)) )
for k, col in zip(unique_labels, colors):
    if k == -1:
        col = 'k'
    class_member_mask = (labels == k)
    #Plotting clustered datapoints
    xy = X[class_member_mask & core_samples_mask]
    plt.scatter(xy[:,0], xy[:,1], s=50, c=col, marker=u'o', alpha=0.5)
    #Plotting Outliers
    xy = X[class_member_mask & ~core_samples_mask]
    plt.scatter(xy[:,0], xy[:,1], s=50, c=col, marker=u'o', alpha=0.5)
    plt.savefig("DBS_nodata.pdf")

