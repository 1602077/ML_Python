""" Polynomial Regression - CO2 Emission Dataset """
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from IPython.display import display

#Data Intialisation
df = pd.read_csv("FuelConsumption.csv")
#display(df.head())
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#display(cdf.head(9))

#plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
#plt.xlabel("Engine size")
#plt.ylabel("Emission")
#plt.show()

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]

from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])


poly = PolynomialFeatures(degree=2)
#degree specifies the degree of the poly i.e. 2 is quadratic
train_x_poly = poly.fit_transform(train_x)
#fit_transform takes x values and outputs a list of our data raised from power of 0 to degree specified. Polynomial regression is a special case of linear regression - the user simply choose the number of features via poly deg. We can therefore transform this is a linear regressino problem.
train_x_poly

clf = linear_model.LinearRegression()
train_y_ = clf.fit(train_x_poly,train_y)
print('Coeff.: ', clf.coef_)
print('Int.: ', clf.intercept_)
#Coeff and Int are parameters of the fitted line. This is a typical linear regression usnig three parameters and knowing the parameters are the intercepts and coefficents of hyperlane skleanr has estimate them for the new set of features.

#plotting
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

#Evaluation

from sklearn.metrics import r2_score
test_x_poly = poly.fit_transform(test_x)
test_y_ = clf.predict(test_x_poly)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y_ , test_y)) 

#Q: NOW ATTEMPT TO PERFORM A REGRESSION WITH DEGREE OF THREE - IS IT MORE ACCURATE?
clf3 = linear_model.LinearRegression()
train_x_poly3 = PolynomialFeatures(degree=3).fit_transform(train_x)
train_y3 = clf3.fit(train_x_poly3, train_y)
print('Coeff.: ', clf3.coef_)
print('Int.: ', clf3.intercept_)
#Plotting
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS,  color='blue')
XX3 = np.arange(0.0, 10.0, 0.1)
yy3 = clf3.intercept_[0]+ clf3.coef_[0][1]*XX3+ clf3.coef_[0][2]*np.power(XX3, 2)+ clf3.coef_[0][3]*np.power(XX3, 3)
plt.plot(XX3, yy3, '-r' )
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()
#Evaulation
test_x_poly3 = PolynomialFeatures(degree=3).fit_transform(test_x)
test_y3_ = clf3.predict(test_x_poly3)
print("Mean absolute error: %.2f" % np.mean(np.absolute(test_y3_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y3_ - test_y) ** 2))
print("R2-score: %.2f" % r2_score(test_y3_ , test_y) )
