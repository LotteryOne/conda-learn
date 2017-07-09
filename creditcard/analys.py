from matplotlib import pyplot as plt
import pandas as pd

credit = pd.read_csv('E:/python/tf/creditcard/creditcard.csv')
print(credit.shape)
print(credit.columns)
print(credit.head(1))
pd.value_counts(credit['Class'],sort=True).sort_index()



