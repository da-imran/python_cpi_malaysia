# Predictive Analysis on Cost of Living in Malaysia
# Predict the CPI value for the next (1,2,3...) months
# First ML Model - Linear Regression

#Features selection multiple attribute / all methods
#Multiple Linear Regression....

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics as met
import matplotlib.pyplot as plt

df = pd.read_excel('Predictive Analysis Cost of Living Malaysia\Python\cpi.xlsx')
dfLabel = df.copy()
'''
# Visualise data
plt.figure(figsize=(18,8))
plt.title('Total CPI per Month')
plt.xlabel('Months')
plt.ylabel('CPI')
plt.plot(df.loc[:,'Total'])
plt.show()
'''
#Get the data required data to be stored
df = df[['Health']]

#Months to predict
predictMonths = 12

#New column (target) shifted 'x' units up
df['Prediction'] = df[['Health']].shift(-predictMonths)

#Create feature data set (X), convert numpy, remove last 'x' rows/month
X = np.array(df.drop(['Prediction'],1))[:-predictMonths]

#Create target data set (Y), convert numpy, get all target values except last 'x' rows/month
Y = np.array(df['Prediction'])[:-predictMonths]

#Split 75% training, 25% test
xTrain, xTest, yTrain, yTest = train_test_split(X,Y,test_size= 0.30)

#Create model
#Linear Regression
lr = LinearRegression().fit(xTrain,yTrain)

#Retrieve last 'x' rows of feature data set
xFuture = df.drop(['Prediction'],1)[:-predictMonths]
xFuture = xFuture.tail(predictMonths)
xFuture = np.array(xFuture)

#Show linear model prediction
lrPrediction = lr.predict(xFuture)

#Testing the model
linearAccuracy = lr.score(xTest,yTest)
linearMAE = met.mean_absolute_error(xTest,yTest)
linearMSE = met.mean_squared_error(xTest,yTest)
linearR2 = met.r2_score(xTest,yTest)

print('Linear Regression Test Result')
print(' Accuracy \t MAE \t MSE \t R-squared \t')
print(' {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(linearAccuracy,linearMAE,linearMSE,linearR2))
print()

actVal = np.array(df['Health'].tail(12))

print('Prediction value \n',lrPrediction)
print('Actual value \n',actVal)

#Visualise the data
predictions = lrPrediction
validData = df[X.shape[0]:]
validData['Prediction'] = predictions
plt.figure(figsize=(18,10))
plt.title('Linear Model Prediction')
plt.xlabel('Months')
plt.ylabel('Health CPI')
plt.plot(df['Health'])
plt.plot(validData[['Health', 'Prediction']])
plt.legend(['Original','Val','Pred'])
plt.show()

validData2 = df[X.shape[0]:]
validData2['Prediction'] = predictions
plt.figure(figsize=(18,10))
plt.title('Linear Model Prediction')
plt.xlabel('Months')
plt.ylabel('Health CPI')
plt.plot(validData[['Health', 'Prediction']])
plt.legend(['Original','Prediction'])
plt.show()
