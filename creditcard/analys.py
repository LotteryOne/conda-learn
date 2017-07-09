from matplotlib import pyplot as plt
import pandas as pd

#
credit = pd.read_csv('E:/python/tf/creditcard/creditcard.csv')
###first see what like the data
# print(credit.shape)
# print(credit.columns)
# print(credit.head(1))
# count_class = pd.value_counts(credit['Class'], sort=True).sort_index()
# count_class.plot(kind='bar')
# plt.title('fund class histogram')
# plt.xlabel('Class')
# plt.ylabel('frequency')
# plt.show()

from sklearn.preprocessing import StandardScaler
import numpy  as np

####transfrom column  data
credit['normAmount'] = StandardScaler().fit_transform(credit['Amount'].reshape((-1, 1)))
credit = credit.drop(['Time', 'Amount'], axis=1)
# print(credit.head(1))

X = credit.ix[:, credit.columns != 'Class']
Y = credit.ix[:, credit.columns == 'Class']

num_fraud_reco = len(credit[credit.Class == 1])
fraud_index = np.array(credit[credit.Class == 1].index)
norm_index = credit[credit.Class == 0].index

random_normal_index = np.random.choice(norm_index, num_fraud_reco, replace=False)
random_normal_index = np.array(random_normal_index)

# print(len(fraud_indec), ',', type(fraud_indec), '::', len(random_normal), ',', type(random_normal))
under_sample_index = np.concatenate([random_normal_index, fraud_index])
under_sample_data = credit.iloc[under_sample_index, :]

# X_undersample = under_sample_data.ix[:, under_sample_data.column != 'Class']
# Y_undersample = under_sample_data.ix[:, under_sample_data.column == 'Class']

print('nomal data:', len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
print('fraul data:', len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
print('total data:', len(under_sample_data))
