# Predictive Analysis on Cost of Living in Malaysia
# Predict the CPI value for the next (1,2,3...) months
# Second ML Model - Decision Tree Regression

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
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
predictMonths = 24

#New column (target) shifted 'x' units up
df['Prediction'] = df[['Health']].shift(-predictMonths)

#Create feature data set (X), convert numpy, remove last 'x' rows/month
X = np.array(df.drop(['Prediction'],1))[:-predictMonths]

#Create target data set (Y), convert numpy, get all target values except last 'x' rows/month
Y = np.array(df['Prediction'])[:-predictMonths]

#Split 75% training, 25% test
xTrain, xTest, yTrain, yTest = train_test_split(X,Y,test_size=0.30)

#Create model
#Decision Tree Regression
tree = DecisionTreeRegressor().fit(xTrain,yTrain)

#Retrieve last 'x' rows of feature data set
xFuture = df.drop(['Prediction'],1)[:-predictMonths]
xFuture = xFuture.tail(predictMonths)
xFuture = np.array(xFuture)

#Show tree model prediction
treePrediction = tree.predict(xFuture)

#Testing the model
treeAccuracy = tree.score(xTest,yTest)
treeMAE = met.mean_absolute_error(xTest,yTest)
treeMSE = met.mean_squared_error(xTest,yTest)
treeR2 = met.r2_score(xTest,yTest)


print('Decision Tree Regression Metric Result')
print('Accuracy \t MAE \t MSE \t R-squared')
print('{:.4f} \t\t {:.4f}  {:.4f}  {:.4f}'.format(treeAccuracy,treeMAE,treeMSE,treeR2))
print()

#Visualise the data
predictions = treePrediction
validData = df[X.shape[0]:]
validData['Prediction'] = predictions
plt.figure(figsize=(18,10))
plt.title('Tree Model Prediction')
plt.xlabel('Months')
plt.ylabel('Health CPI')
plt.plot(df['Health'])
plt.plot(validData[['Health', 'Prediction']])
plt.legend(['Original','Val','Pred'])
plt.show()

validData2 = df[X.shape[0]:]
validData2['Prediction'] = predictions
plt.figure(figsize=(18,10))
plt.title('Tree Model Prediction')
plt.xlabel('Months')
plt.ylabel('Health CPI')
plt.plot(validData[['Health', 'Prediction']])
plt.legend(['Original','Prediction'])
plt.show()
