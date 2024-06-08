#Features selection multiple attribute / all methods
#Multiple Linear Regression....

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn import metrics as met
import matplotlib.pyplot as plt

df = pd.read_excel('Predictive Analysis Cost of Living Malaysia\Python\cpi.xlsx')
dfLabel = df.copy()

x = df[['Food & Non-Alcoholic Beverages',
       'Alcoholic Beverages & Tobacco', 'Clothing and Footwear',
       'Housing, Water, Electricity, Gas & Other Fuels',
       'Furnishings, Household Equipment & Routine Household Maintenance',
       'Health', 'Communication', 'Recreation Services & Culture',
       'Education', 'Restaurant and Hotels', 'Miscellaneous Goods & Services']]

y = df[['Total']]

mlr = LinearRegression()
mlr.fit(x,y)
predMLR = mlr.predict(x)

print('Actual value \n',np.array(y))
print('Predicted value \n',predMLR)
yTrue = np.array(y)
yPred = predMLR

#Model Evaluation
r2 = mlr.score(x,y)
mae = met.mean_absolute_error(yTrue,yPred)
mse = met.mean_squared_error(yTrue,yPred)
sqrt = np.sqrt(mse)

print('Multiple Linear Regression Evaluation Results')
print('R2 \t\t MAE \t MSE \t Root MSE')
print('{:.4f} \t {:.4f}  {:.4f}  {:.4f}'.format(r2,mae,mse,sqrt))
print()

# Visualise the models
plt.figure(figsize=(18,8))
plt.title('Multiple Linear Regression')
plt.plot(predMLR, color='red', label='Predicted value')
plt.plot(y, color='blue', label='Actual value')
plt.xlabel('Months')
plt.ylabel('Total CPI')
plt.legend()
plt.show()

predMLRdata = pd.DataFrame(predMLR)
fName = "MlrPredData.xlsx"
predMLRdata.to_excel(fName)

