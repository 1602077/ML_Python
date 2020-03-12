""" KNN - Telecommunications Dataset
KNN is a supervised learnig algorithm used for classification, based on the correpsonding points of a given data points. Once a point is to be predicted, it take into account the 'k' nearest points to determine its classification. """
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import itertools
from matplotlib.ticker import NullFormatter
import matplotlib.ticker as ticker
from sklearn import preprocessing
from IPython.display import display

df = pd.read_csv("teleCust1000t.csv")
#display(df.head())
#Contains segemented data for its customer base by service usage patterns, categorising them based on deographic data. The target field - custcat - has four possible values according to the four possible customer services. Objective is to build a classfier that predicts the class of unkown cases using a KNN algorithm.
#display(df['custcat'].value_counts())
# 281 plus service (3), 266 basic service (1), 236 total service (4), and 217 E service (4).

#Use sklearn convert pandas data frame to a numpy array, then standarise data to get zero mean and unit variance - this is good practice.
X = df[['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']] .values  #.astype(float)
X[0:5]
y = df['custcat'].values
y[0:5]
X = preprocessing.StandardScaler().fit(X).transform(X.astype(float))
X[0:5]

#Train Test Split - inc. out of sample accuracy
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
#print ('Train set:', X_train.shape,  y_train.shape)
#print ('Test set:', X_test.shape,  y_test.shape)

from sklearn.neighbors import KNeighborsClassifier
k = 10
#Train Model and Predict
neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
neigh
yhat = neigh.predict(X_test)
yhat[0:5]

#Accuracy Evaluation
from sklearn import metrics
#print("Train set Accuracy: ", metrics.accuracy_score(y_train, neigh.predict(X_train)))
#print("Test set Accuracy: ", metrics.accuracy_score(y_test, yhat))

#Accuracy of KNN for different Ks
Ks = 10
mean_acc = np.zeros((Ks-1))
std_acc = np.zeros((Ks-1))
ConfustionMx = [];
for n in range(1,Ks):
    #Train Model and Predict
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])
mean_acc

plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.legend(('Accuracy ', '+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbours (K)')
plt.tight_layout()
plt.show()
print("The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1)

