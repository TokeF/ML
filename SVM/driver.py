import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import sys
# import SVM_1
from sklearn import preprocessing
sys.path.insert(0, '../utilities')
import PC_anal
import data_visualize

#load data
from data_reader import load_data2
fname = "../data/20171101_RAW_export.xyz"
_ , dbdt, lbl, timestamp = load_data2(fname, 8, 20, 14)

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
