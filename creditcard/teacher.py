import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('E:/python/tf/creditcard/creditcard.csv')

data['normAmount'] = StandardScaler().fit_transform(data['Amount'].reshape(-1, 1))
data = data.drop(['Time', 'Amount'], axis=1)
data.head()

number_records_fraud = len(data[data.Class == 1])
fraud_indices = np.array(data[data.Class == 1].index)

# Picking the indices of the normal classes
normal_indices = data[data.Class == 0].index

# Out of the indices we picked, randomly select "x" number (number_records_fraud)
random_normal_indices = np.random.choice(normal_indices, number_records_fraud, replace=False)
random_normal_indices = np.array(random_normal_indices)

# Appending the 2 indices
under_sample_indices = np.concatenate([fraud_indices, random_normal_indices])

# Under sample dataset
under_sample_data = data.iloc[under_sample_indices, :]

X_undersample = under_sample_data.ix[:, under_sample_data.columns != 'Class']
y_undersample = under_sample_data.ix[:, under_sample_data.columns == 'Class']

# Showing ratio
print("Percentage of normal transactions: ",
      len(under_sample_data[under_sample_data.Class == 0]) / len(under_sample_data))
print("Percentage of fraud transactions: ",
      len(under_sample_data[under_sample_data.Class == 1]) / len(under_sample_data))
print("Total number of transactions in resampled data: ", len(under_sample_data))

from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.metrics import confusion_matrix,recall_score,classification_report
def printing_Kfold_scores(x_train_data,y_train_data):
      fold = KFold(len(y_train_data),5,shuffle=False)

      # Different C parameters
      c_param_range = [0.01,0.1,1,10,100]

      results_table = pd.DataFrame(index = range(len(c_param_range),2), columns = ['C_parameter','Mean recall score'])
      results_table['C_parameter'] = c_param_range

      # the k-fold will give 2 lists: train_indices = indices[0], test_indices = indices[1]
      j = 0
      for c_param in c_param_range:
            print('-------------------------------------------')
            print('C parameter: ', c_param)
            print('-------------------------------------------')
            print('')

            recall_accs = []
            for iteration, indices in enumerate(fold,start=1):

                  # Call the logistic regression model with a certain C parameter
                  lr = LogisticRegression(C = c_param, penalty = 'l1')

                  # Use the training data to fit the model. In this case, we use the portion of the fold to train the model
                  # with indices[0]. We then predict on the portion assigned as the 'test cross validation' with indices[1]
                  lr.fit(x_train_data.iloc[indices[0],:],y_train_data.iloc[indices[0],:].values.ravel())

                  # Predict values using the test indices in the training data
                  y_pred_undersample = lr.predict(x_train_data.iloc[indices[1],:].values)

                  # Calculate the recall score and append it to a list for recall scores representing the current c_parameter
                  recall_acc = recall_score(y_train_data.iloc[indices[1],:].values,y_pred_undersample)
                  recall_accs.append(recall_acc)
                  print('Iteration ', iteration,': recall score = ', recall_acc)

            # The mean value of those recall scores is the metric we want to save and get hold of.
            results_table.ix[j,'Mean recall score'] = np.mean(recall_accs)
            j += 1
            print('')
            print('Mean recall score ', np.mean(recall_accs))
            print('')

      best_c = results_table.loc[results_table['Mean recall score'].idxmax()]['C_parameter']

      # Finally, we can check which C parameter is the best amongst the chosen.
      print('*********************************************************************************')
      print('Best model to choose from cross validation is with C parameter = ', best_c)
      print('*********************************************************************************')

      return best_c