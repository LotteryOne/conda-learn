import pandas as pd, numpy as np

titan_file = 'E:/python/tf/pandas/titanic_train.csv'
titan_sur = pd.read_csv(titan_file)
# print(titan_sur.head(10))
# print(titan_sur.columns)
# age=titan_sur['Age'].loc[0:10]

age = titan_sur['Age']
age_isnull = pd.isnull(age)
age_nulltrue = age[age_isnull]
print(age_nulltrue)
# print(len(age_nulltrue))
# avg = sum(age) / len(age)
# print(avg)
# good_age=age[age_isnull==False]
# print(good_age)

print(age.mean())





