import pandas as pd
import numpy as np

dates = pd.date_range('20180101',periods=6)
df = pd.DataFrame(np.random.randn(6,4),index=dates,columns=['A','B','C','D'])
print(df)

print(df['A'])
print(df[0:3])
#print(df['A','C'])#wrong
#print(df['A':])#wrong format



print(df.loc['2018-01-01'])
print(df.loc['2018-01-01'][:2])
print()
print(df.loc['2018-01-01'],['A','B'])#warning,not like you think
print(df.loc['2018-01-01'],['A'])#warning,not like you think
print(df.loc['2018-01-01',['A','B']])#diffent format and different index

print(df)
#print(df.iloc[0])
#print(df.iloc[2])
print(df.iloc[0,2])#0th row and 2th column
print(df.iloc[0:1,1:3])#not include the second index!!!!!!!!!!!!!!!!!!
print(df.iloc[[0,1],[1,3]])#

print(df.ix[:3,['A','C']])
print('-----------------------')
print('df[df.A>0]:',df[df.A>0])
print('split to analyse')
print('df.A>0:',df.A>0)
print(type(df.A>0))
print((df.A>0).shape)
boolean_array = pd.Series([False,True,False,True,False,True],index=dates)
print('boolean_array:\n',boolean_array)
print(boolean_array.shape)
print(type(boolean_array))

#######error1#############can't be a <class 'pandas.core.series.Series'>
########error2:Unalignable boolean Series provided as indexer###index 2018-01-01 and 0 is different....change index!!!
print(df[boolean_array])





