import pandas

file1 = "E:/python/tf/pandas/food_info.csv"

food_info = pandas.read_csv(file1)
print(type(food_info))
# print(food_info.dtypes)
# print(help(pandas.read_csv))

# print(food_info.head())
# print(food_info.tail(4))

###see data_info
print(food_info.columns)
# print(food_info.shape)
# print(food_info.loc[0])

# columsl = ['NDB_No', 'Shrt_Desc']
# ndb_col = food_info[columsl]

# print(ndb_col)



# collist = food_info.columns.tolist()
# print(collist)
# g_columns = []
# for c in collist:
#     if c.endswith("(g)"):
#         g_columns.append(c)
#
# print(g_columns)

# print(food_info['Water_(g)']/1000)

# max_calors = food_info['Energ_Kcal'].max()
# print(max_calors)

# print(food_info['Protein_(g)']/food_info['Protein_(g)'].max())

# food_info.sort_values('Sodium_(mg)', inplace=True, ascending=False)
# print(food_info['Sodium_(mg)'])


