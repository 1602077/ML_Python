""" Logistic Regression - Telecommunications Dataset """
import numpy as np
import pandas as pd
from IPython.display import display

churn_df = pd.read_csv("ChurnData.csv")
#Selection of categories to use and then changing target data type to be integer as required by sklearn.
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
display(churn_df.head())
#display(churn_df.shape) 200 rows x 10 columns

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
display(X[0:5])
y = np.asarray(churn_df['churn'])
display(y[0:5])

from sklearn import preprocessing
X = preprocessing.StandardScaler().fit(X).transform(X) #normalising
display(X[0:5])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

#Building of logistic regression model - type of model used in sklearn support regularisation, this solves problem of overfitting of data. C parameter indicates inverse regularisaition strength (must be a positive float) - smaller value = stronger regularisation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
yhat = LR.predict(X_test)
display(yhat)

#predict_proba return estimates for all classes ordered by class label. Col. 1 = prob of class 1 P(Y=1|x), Col.2 = prob of class 0 P(Y=0|X)

yhat_prob = LR.predict_proba(X_test)
display(yhat_prob)

#Evaluation of model accuracy using jaccard index - defined as the size of the intersection divided by the size of the union of two label sets. If the entire set of predicted labels for a sample strictly match with the true set of labels, then the subset accuracy is 1.0; otherwise it is 0.0.

from sklearn.metrics import jaccard_similarity_score
print(jaccard_similarity_score(y_test, yhat))

#Confusion Matrix - another method to look at the accuracy of the classifier
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import matplotlib.pyplot as plt
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

print (classification_report(y_test, yhat))
#Precision is a measure of the accuracy provided that a class label has been predicted. It is defined by: precision = TP / (TP + FP) Recall is true positive rate. It is defined as: Recall = TP / (TP + FN). F1 score is the harmonic average of the precision and recall, where an F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0. It is a good way to show that a classifer has a good value for both recall and precision.

#Log loss
from sklearn.metrics import log_loss
print ("Log loss: ", log_loss(y_test, yhat_prob))

