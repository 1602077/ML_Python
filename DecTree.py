""" Decision Trees - Drug Classification Dataset """
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from IPython.display import display

df = pd.read_csv("drug200.csv", delimiter=",")
df[0:5]
#display(df.head())
#display(df.shape)

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
X[0:5]
#Feature like sex or BP are categorical and must be converted to numerical values using pandas.get_dummies()

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])


le_Chol = preprocessing.LabelEncoder()
le_Chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_Chol.transform(X[:,3])

X[0:5]
y = df["Drug"]
y[0:5]
#Test Train & Split
from sklearn.model_selection import train_test_split
#train_test_split takes X, y, test_size=0.3, and random_state=3 and returns X_trainset, X_testset, y_trainset, y_testset 
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

#Modelling
#Create an instance of DecisionTreeClassifier called drugTree, specifiying criterion="entropy" allows one to see the information gain at each node

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree # it shows the default parameters
#fit data with training fature matrix X_trainset and training response vector y_trainset
drugTree.fit(X_trainset,y_trainset)

#Predictions on the testing dataset
predTree = drugTree.predict(X_testset)
#print (predTree [0:5])
#print (y_testset [0:5])

#Evaluation - checking accuracy of model
from sklearn import metrics
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, predTree))

#Visualisation of Decision Tree
from sklearn.externals.six import StringIO
import matplotlib.pyplot as plt
import pydotplus
import matplotlib.image as mpimg
from sklearn import tree

dot_data = StringIO()
filename = "drugtree.png"
featureNames = df.columns[0:5]
targetNames = df["Drug"].unique().tolist()
out=tree.export_graphviz(drugTree,feature_names=featureNames, out_file=dot_data, class_names= np.unique(y_trainset), filled=True,  special_characters=True,rotate=False)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png(filename)
img = mpimg.imread(filename)
plt.figure(figsize=(100, 200))
plt.imshow(img,interpolation='nearest')

