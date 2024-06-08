# Predictive Analysis on Cost of Living in Malaysia
# Find and sort the top increase of CPI per months/year

import numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_excel('Predictive Analysis Cost of Living Malaysia\Python\percentageChangeCPI.xlsx')
tempdf = df
# To check the index of the months
# print(df.loc[:,'Months'])

#Test between months based on the index i.e. 0 = Jan-Feb 2011
testMth = 11

temp = df.loc[testMth, df.columns != 'Months',]
df2 = temp.to_frame()

#Sort the variables to find the highest to lowest CPI percentage increase
sortDf = df2.sort_values(by=testMth, ascending=False)
print(sortDf)

rankDf = sortDf.rank(ascending = False)
print(rankDf)
