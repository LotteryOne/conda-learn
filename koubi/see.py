import pandas as pd
from matplotlib import pyplot as pt
import numpy as np

koubi = 'E:/python/tf/koubi/data.csv'

k_cols = pd.read_csv(koubi)
print(k_cols.columns)
print(k_cols.shape)
print(k_cols.head())
row = k_cols[pd.notnull(k_cols['shot_made_flag'])]

alpha = 0.2
pt.figure(figsize=(10, 10))
pt.subplot(121)
pt.scatter(row.loc_x, row.loc_y, color='R', alpha=alpha)
pt.title('see koubi sport data')

pt.subplot(122)
pt.scatter(row.lon, row.lat, color='B', alpha=alpha)
pt.title('other way')

pt.show()

# fig.scatter(row.loc_x,row.loc_y,color='R', alpha=alpha)
# fig.title('see config')
# fig.show()


