import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def load_data(fileName): 
    #read .xyz datafile into pandas data frame
    dataFrame = pd.read_csv(fileName, sep='\s+',header=[3])

    #correct header names
    #first column is a \ due to data file. Thus labels are shifted one back. Column 179 is empty and thus deleted
    header = dataFrame.columns.drop('/')
    dataFrame.drop("DBDT_INUSE_40", axis=1, inplace=True)
    dataFrame.columns = header
    return dataFrame


# fname = "20171121CH2_export.xyz"
# df = load_data(fname)

# # working on visualization tool
# a = df.loc[:,'DBDT_1':'DBDT_20'].values
# lbl = df.loc[:,'DBDT_INUSE_14'].values
# to = 10000
# x = range(to)
# #Normalize
# a = (a - np.amin(a)) / (np.amax(a) - np.amin(a))

# # colors = ['red','green','blue','purple']
# # plt.plot(x, lbl[0:to])
# for i in range(20):
#     plt.plot(x,a[0:to,i])
#     # plt.scatter(x,a[0:to,i], c=lbl[0:to], cmap= matplotlib.colors.ListedColormap(colors))
# # plt.ylim(np.amin(a),np.amax(a))
# plt.show()