import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
# import SVM_1
from sklearn import preprocessing
#sys.path.insert(0, '../utilities')
from utilities import data_visualize, difference
from utilities.data_reader import load_data2

fname = "../data/20171101_RAW_export.xyz"
df, dbdt, lbl, timestamp = load_data2(fname, 8, 8)

df2 = pd.DataFrame({'a':[1, 2, 3, 4, 5, 6],
                   'b':[4, 5, 6, 7, 8, 9],
                   'c':[7, 8, 9, 10, 11, 12]})
b = difference.mvg_avg(df, 11)
data_visualize.plotDat(timestamp, b, lbl)
plt.yscale('log')
plt.show()
def plotNormalnratio() :
    fname = "../data/20171101_RAW_export.xyz"
    _ , dbdt, lbl, timestamp = load_data2(fname, 8, 20)

    ratio = difference.row_ratio(timestamp, dbdt)
    data_visualize.plotDat(timestamp, ratio, lbl)
    data_visualize.plotDat(timestamp, dbdt, lbl)
    plt.yscale('log')
    plt.show()
# PC_anal.PCA_custom(dbdt, lbl, 'hej')
# exit()
# #normalize (min max)
# print("Min-max norm:")
# X_minmax = (dbdt - np.amin(dbdt)) / (np.amax(dbdt) - np.amin(dbdt))
# SVM_1.SVM_classify(X_minmax,lbl)
# print("")

# #standardize (z-score)
# dbdt_scaler = preprocessing.StandardScaler().fit(dbdt)
# dbdt_std = dbdt_scaler.transform(dbdt)
# X_std = dbdt_std
# print("z score norm:")
# SVM_1.SVM_classify(X_std,lbl)
# print("")




# # TESTing the plot
# print(dbdt.shape)
# print(lbl.shape)
# dbdt = np.array([[1, 2], [1, 2], [1, 2],[1, 2],[1, 2],[1, 2]])
# print(dbdt.shape)
# lbl= np.array([1, 0, 1, 1, 1, 1])
# print(lbl.shape)
# timestamp = np.array([[1], [2], [3], [4], [5], [6]])