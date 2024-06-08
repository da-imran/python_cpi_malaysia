# Predictive Analysis on Cost of Living in Malaysia
# Predict the CPI value for the next (1,2,3...) months
# Third ML Model - SVR (Support Vector Regression Machine)

#Multiple SVM Variable

import numpy as np
import pandas as pd
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn import metrics as met

df = pd.read_excel('Predictive Analysis Cost of Living Malaysia\Python\cpi.xlsx')
dfLabel = df.copy()

#Store last 12 months of (Total)data
predict_cpi = df.loc[0:108, 'Health']

#Store all data except last 12 months of (Total)data
allData = df.loc[0:108, 'Health']

#Save last 12 Total CPI into list
prdMnth = list()
for i in df['Health'].tail(12):
    prdMnth.append(i)

#Empty list months and cpi
lstMonth = list()
months = list()
cpi = list()

#Create independent data set (months)
for mth in range(109):
    months.append([int(mth)])

for lstMnth in range(12):
    lstMonth.append([int(lstMnth)])

#Create dependent data set (cpi)
for value in predict_cpi:
    cpi.append(float(value))

#3 models using SVM
lin_svr = SVR(kernel='linear', C=100.0)
lin_svr.fit(months,cpi)

poly_svr = SVR(kernel='poly', C=100.0, degree=2)
poly_svr.fit(months,cpi)

rbf_svr = SVR(kernel='rbf', C=100.0, gamma=0.85)
rbf_svr.fit(months,cpi)

#Get each number of months to be predicted
month = [109],[110],[111],[112],[113],[114],[115],[116],[117],[118],[119],[120]

#Get all predicted value
rbfResult = rbf_svr.predict(month)
linResult = lin_svr.predict(month)
polyResult = poly_svr.predict(month)

print('The RBF SVR predicted : ',rbfResult)
print()
print('The Linear SVR predicted : ',linResult)
print()
print('The Polynomial SVR predicted : ',polyResult)
print()

answer = list()
for i in df['Health'].tail(12):
    answer.append(i)
print('Actual values', answer,'\n')

#Statistic data result
linAcc = lin_svr.score(months,cpi)
linMAE = met.mean_absolute_error(answer,linResult)
linMSE = met.mean_squared_error(answer,linResult)
linRMSE = np.sqrt(linMSE)
linR2 = met.r2_score(answer,linResult)

polyAcc = poly_svr.score(months,cpi)
polyMAE = met.mean_absolute_error(answer,polyResult)
polyMSE = met.mean_squared_error(answer,polyResult)
polyRMSE = np.sqrt(polyMSE)
polyR2 = met.r2_score(answer,polyResult)

rbfAcc = rbf_svr.score(months,cpi)
rbfMAE = met.mean_absolute_error(answer,rbfResult)
rbfMSE = met.mean_squared_error(answer,rbfResult)
rbfRMSE = np.sqrt(rbfMSE)
rbfR2 = met.r2_score(answer,rbfResult)

print('RBF Result')
print('Accuracy \t MAE \t MSE \t RSME \t R-squared')
print('{:.4f} \t\t {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(rbfAcc,rbfMAE,rbfMSE,rbfRMSE,rbfR2))
print()

print('Linear Result')
print('Accuracy \t MAE \t MSE \t RSME \t R-squared')
print('{:.4f} \t\t {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(linAcc,linMAE,linMSE,linRMSE,linR2))
print()

print('Polynomial Result')
print('Accuracy \t MAE \t MSE \t RSME \t R-squared')
print('{:.4f} \t\t {:.4f}  {:.4f}  {:.4f}  {:.4f}'.format(polyAcc,polyMAE,polyMSE,polyRMSE,polyR2))
print()

# Visualise the predicted value on all models
plt.figure(figsize=(18,8))
plt.title('Health CPI predicted for 12 month')
plt.scatter(month, prdMnth, color = 'black', label = 'Actual data')
plt.plot(month, rbfResult, color = 'green', label = 'RBF Prediction')
plt.plot(month, linResult, color = 'red', label = 'Linear Prediction')
plt.plot(month, polyResult, color = 'blue', label = 'Polynomial Prediction')
plt.xlabel('Months')
plt.ylabel('Health CPI')
plt.legend()
plt.show()

# Visualise the models
plt.figure(figsize=(18,8))
plt.title('Health CPI per month')
plt.scatter(months, cpi, color = 'black', label = 'Data')
plt.plot(months, rbf_svr.predict(months), color = 'green', label = 'RBF Model')
plt.plot(months, lin_svr.predict(months), color = 'red', label = 'Linear Model')
plt.plot(months, poly_svr.predict(months), color = 'blue', label = 'Polynomian Model')
plt.xlabel('Months')
plt.ylabel('Health CPI')
plt.legend()
plt.show()
