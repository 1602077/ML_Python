""" Multiple Regression Model - CO2 Emissions Dataset """
import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np
from IPython.display import display

"""" DATA INTIALISATION """

df=pd.read_csv("FuelConsumption.csv")
#display(df.head())
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
#display(cdf.head(9))

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS,  color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.savefig("fig1.pdf")
#plt.show()

""" TRAIN & TEST """

msk = np.random.rand(len(df)) < 0.8
train = cdf[msk]
test = cdf[~msk]
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.savefig("fig2.pdf")

from sklearn import linear_model
regr = linear_model.LinearRegression()
x = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit (x, y)
print('Coefficients: ', regr.coef_)

#Coefficeints are calculated using ordinary least squares (OLS) methods - this minimises the sum of the squares of the differences between the target variable and that predicted by the linear function.

""" Prediction """

y_hat= regr.predict(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
x = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB']])
y = np.asanyarray(test[['CO2EMISSIONS']])
print("Residual sum of squares: %.2f"
      % np.mean((y_hat - y) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % regr.score(x, y))

