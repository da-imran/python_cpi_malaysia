#Features selection multiple attribute / all methods
#Multi-output SVM(Regression)....

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR
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

svr = SVR(epsilon=0.2)
mor = MultiOutputRegressor(svr)
mor = mor.fit(x,y)
predMsvm = mor.predict(x)

print('Actual value \n',np.array(y))
print('Predicted value \n',predMsvm)
yTrue = np.array(y)
yPred = predMsvm

#Model Evaluation
r2 = mor.score(x,y)
mae = met.mean_absolute_error(yTrue,yPred)
mse = met.mean_squared_error(yTrue,yPred)
sqrt = np.sqrt(mse)

print('Simple Vector Regression Machine Evaluation Results')
print('R2 \t\t MAE \t MSE \t Root MSE')
print('{:.4f} \t {:.4f}  {:.4f}  {:.4f}'.format(r2,mae,mse,sqrt))
print()

# Visualise the models
plt.figure(figsize=(18,8))
plt.title('Simple Vector Regression Machine')
plt.plot(predMsvm, color='red', label='Predicted value')
plt.plot(y, color='blue', label='Actual value')
plt.xlabel('Months')
plt.ylabel('Total CPI')
plt.legend()
plt.show()

