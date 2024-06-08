import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#Features selection for predictive analytics on cost of living in Malaysia
df = pd.read_excel('Predictive Analysis Cost of Living Malaysia\Python\cpi.xlsx')
dfLabel = df.copy()

x = df[['Food & Non-Alcoholic Beverages',
       'Alcoholic Beverages & Tobacco', 'Clothing and Footwear',
       'Housing, Water, Electricity, Gas & Other Fuels',
       'Furnishings, Household Equipment & Routine Household Maintenance',
       'Health', 'Transport', 'Communication', 'Recreation Services & Culture',
       'Education', 'Restaurant and Hotels', 'Miscellaneous Goods & Services']]

y = df[['Total']]

#Pearson Correlation
plt.figure(figsize=(20,12))
cor = df.corr()
sns.heatmap(cor, annot=True, cmap=plt.cm.Reds)
plt.show()

#Correlation output variable
corTarget = abs(cor["Total"])

#Select highly correlated features
relevantFt = corTarget[corTarget>0.4]
print(relevantFt)

