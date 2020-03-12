""" Suport Vector Machines (SVM) - Cancer Dataset """
import numpy as np
import pandas as pd
from IPython.display import display
import matplotlib.pyplot as plt
cell_df = pd.read_csv("cell_samples.csv")
display(cell_df.head(5))
#Characteristics of cell samples  are graded from 1 to 10, with 1 being closest to benign. 
#Class field contains diagnosis (confirmed by a seperate procedure: benign =2 or malignant = 4.
#Look at distributino of classes based on clump thickness and uniformity of cell size.
#ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color= 'darkblue', label='malignant');
#cell_df[cell_df['Class'] ==2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
#plt.show()

#CONVERSION OF PANDA DF TO NP ARRAY
#BareNuc contains non numerical values, drop these.
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuce'] = cell_df['BareNuc'].astype('int')
feature_df = cell_df[['Clump', 'UnifSize', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit' ]]
X = np.asarray(feature_df)
#print("X: ", X[0:5])
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
print("y: ", y[0:5])

#TRAIN AND TEST DATASET
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
#print('Train set: ', X_train.shape, y_train.shape)
#print('Test Set: ', X_test.shape, y_test.shape)

#MODELLING - SVM USING SKLEARN, Kernelling of data (mapping to a high dim. space)

from sklearn import svm
clf = svm.SVC(kernel='rbf') #radial basis function
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)
print("yhat: ", yhat[0:5]) #predicted values of y bases on kernelling of data

#EVALUATION - Comparing accuracy of yhat compared to y
from sklearn.metrics import classification_report, confusion_matrix
import itertools
def plot_confusion_matrix(cm, classes, normalize=False, title='confusion matrix', cmap=plt.cm.Blues):
    """
    Prints and plots confusion matrix. To normalise set to True.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1) [:, np.newaxis]
        print("Normalised Confusion Matrix")
    else:
        print("Confusion Matrix w/o Normalisation")
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center", color="white"
        if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.xlabel('True Label')
    plt.ylabel('Predicted Label')

#CONFUSION MATRIX
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)
print(classification_report(y_test, yhat))
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)', 'Malignant(4)'], normalize=False, title = 'Confusion matrix')

#F1 SCORE
from sklearn.metrics import f1_score
F1 = f1_score(y_test, yhat, average='weighted')
print('F1: ', F1)

#JACCARD INDEX
from sklearn.metrics import jaccard_similarity_score
JAC_INDX = jaccard_similarity_score(y_test, yhat)
print('Jaccard Index: ', JAC_INDX)

