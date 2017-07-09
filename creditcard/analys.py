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

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
Y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

print('nomal data:', len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
print('fraul data:', len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
print('total data:', len(under_sample_data))

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=0)

print('x_tr,x_te,y_tr,y_te:', len(X_train), len(X_test), len(y_train), len(y_test))

x_train_sample, x_test_sample, y_train_sample, y_test_sample = train_test_split(X_undersample, Y_undersample,
                                                                                test_size=0.3, random_state=0)
print('x_tr,x_te,y_tr,y_te:', len(x_train_sample), len(x_test_sample), len(y_train_sample), len(y_test_sample))

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix, recall_score, classification_report


def printKfolscore(x_train, y_train):
    fold = KFold(len(y_train), 5, shuffle=False)
    c_param_range = [0.01, 0.1, 1, 10, 100]
    result_table = pd.DataFrame(index=range(len(c_param_range), 2), columns=['C_paramter', 'Mean recall score'])
    result_table['C_paramter'] = c_param_range
    j = 0
    for c_param in c_param_range:
        print('=============================')
        print('C_paramter:', c_param)

        recall_accs = []

        for iteration, index in enumerate(fold, start=1):
            lr = LogisticRegression(C=c_param, penalty='l1')
            lr.fit(x_train.iloc[index[0], :], y_train.iloc[index[0], :].values.ravel())

            y_pred_undersample = lr.predict(x_train.iloc[index[1], :].values)

            recall_acc = recall_score(y_train.iloc[index[1], :].values, y_pred_undersample )
            recall_accs.append(recall_acc)
            print('Iteration', iteration, ':recall_code = ', recall_acc)

        result_table.ix[j, 'Mean recall score'] = np.mean(recall_accs)
        j += 1
        print('')
        print('Mean recall score', np.mean(recall_acc))
        print('')
    best_c = result_table.loc[result_table['Mean recall score'].idxmax()]['C_paramter']

    print('base model to choose from cross validation is woth C_parameter:', best_c)

    return best_c


printKfolscore(x_train_sample, y_train_sample)
