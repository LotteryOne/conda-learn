from matplotlib import pyplot as pt
import pandas as pd, numpy as np

rate_file = 'E:/python/tf/plt/UNRATE.csv'
unrate = pd.read_csv(rate_file)
unrate['DATE'] = pd.to_datetime(unrate['DATE'])
print(unrate)
# print(pd.to_datetime(unrate['DATE']))


# pt.plot()
# pt.show()

# first_twelve = unrate[0:12]
# sec_twelve = unrate[12:24]
# sec_twelve['MONTH'] = sec_twelve['DATE'].dt.month
# print(sec_twelve['MONTH'])
# first_twelve['MONTH'] = first_twelve['DATE'].dt.month
# print(first_twelve)
# pt.figure()  # figsize=(6, 3)
# pt.plot(first_twelve['MONTH'], first_twelve['VALUE'], c='red', label='first')
# pt.plot(sec_twelve['MONTH'], sec_twelve['VALUE'], c='blue', label='sec')
# pt.legend(loc='bset')
# pt.xticks(rotation=45)
# pt.xlabel('DATE')
# pt.ylabel('unemployment rate')
# pt.title('first user pyplot')
# pt.show()

# fig = pt.figure(figsize=(6, 3))
# ax1 = fig.add_subplot(3, 2, 1)
# ax1.plot(np.random.randint(1, 5, 5), np.arange(5))
# ax2 = fig.add_subplot(3, 2, 2)
# ax3 = fig.add_subplot(3, 2, 5)
# pt.show()



unrate['MONTH'] = unrate['DATE'].dt.month
unrate['MONTH'] = unrate['DATE'].dt.month
fig = pt.figure(figsize=(6,3))

pt.plot(unrate[0:12]['MONTH'], unrate[0:12]['VALUE'], c='red')
pt.plot(unrate[12:24]['MONTH'], unrate[12:24]['VALUE'], c='blue')
pt.legend(loc='bset')
pt.xticks(rotation=45)
pt.xlabel('DATE')
pt.ylabel('unemployment rate')
pt.title('first user pyplot')
pt.show()