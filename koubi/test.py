from util.UnitUitl import *

min_score = 100000
best_n = 0
scores_n = []
rang_n = np.logspace(0, 2, num=3).astype(int)

# for n in rang_n:
#     print('the number of trees:{0}'.format(n))
#     t1 = time.time()
#     rfc_core = 0
#     rfc = RandomForestRegressor(n_estimators=n)
#     for train_k, test_k in KFold(len(train_kobe), n_folds=10, shuffle=True):
#         rfc.fit(train_kobe.iloc[train_k], train_label.iloc[train_k])
#         # rfc_score += rfc.score(train.iloc[test_k], train_y.iloc[test_k])/10
#         pred = rfc.predict(train_kobe.iloc[test_k])
#         rfc_score += log_loss(train_label.iloc[test_k], pred) / 10
#     scores_n.append(rfc_score)
#     if rfc_score < min_score:
#         min_score = rfc_score
#         best_n = n
#
#     t2 = time.time()
#     print('Done processing {0} trees ({1:.3f}sec)'.format(n, t2 - t1))
# print(best_n, min_score)
