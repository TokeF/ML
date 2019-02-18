import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from utilities import data_visualize, difference, PC_anal
from utilities.data_reader import load_data2
from SVM.rndForest import rnd_forest
# from SVM.lstm_one import lstm_mini
from SVM.SVM_1 import SVM_classify

fname = "../data/20171101_RAW_export.xyz"
# fname = "../data/stendalmark_20181120_RAW_export.xyz"
df, dbdt, lbl, timestamp = load_data2(fname, 8, 23)

slope, rvalue = difference.linear_deriv(timestamp, dbdt, 5)
data_visualize.plotDat(timestamp, slope, lbl)
plt.yscale('log')
plt.title('slope vs Time')
plt.xlim((0.0014+4.30404e4, 0.0028+4.30404e4))
plt.show()
exit()
rvalue = np.square(rvalue)
ratio = difference.row_ratio(timestamp, dbdt)
avg = difference.mvg_avg(pd.DataFrame(dbdt), 11)
var = difference.mvg_var(pd.DataFrame(dbdt), 11)
# super = slope
super = np.concatenate((avg, var, slope, rvalue, ratio), axis=1)
lbl = lbl[~np.isnan(super[:,0])]
super = super[~np.isnan(super[:,0])]
print(super.shape)

# lstm_mini(super, lbl)
# rnd_forest(timestamp, super, lbl, 0.20, 10000)
# PC_anal.PCA_custom(super, lbl, "super")
exit()

avg = difference.mvg_avg(pd.DataFrame(dbdt), 11)
data_visualize.plotDat(timestamp, avg, lbl)
plt.yscale('log')
data_visualize.plotDat(timestamp, avg, lbl)
plt.yscale('log')

#remove NAN values, stemming from windowing
lbl = lbl[~np.isnan(avg[:,0])]
avg = avg[~np.isnan(avg[:,0])]
# rnd_forest(timestamp, dbdt, lbl, 0.30)
# plt.show()
rnd_forest(timestamp, avg, lbl, 0.30)

def plotNormalnratio() :
    fname = "../data/20171101_RAW_export.xyz"
    _ , dbdt, lbl, timestamp = load_data2(fname, 8, 20)

    ratio = difference.row_ratio(timestamp, dbdt)
    data_visualize.plotDat(timestamp, ratio, lbl)
    data_visualize.plotDat(timestamp, dbdt, lbl)
    plt.yscale('log')
    plt.show()

def scatterPlt(x, y):
    # setup figure
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('slope', fontsize=15)
    ax.set_ylabel('r**2', fontsize=15)
    ax.set_title('Scatt: ', fontsize=20)
    # Set target ie. inuse or not, color and shape for data
    targets = [0, 1]
    colors = ['r', 'g']
    shapes = ['o', '^']
    # iterate over a tuple (zip), scatter plot 0 and 1
    for target, color, shape in zip(targets, colors, shapes):
        indicesToKeep = lbl == target
        ax.scatter(x[indicesToKeep]
                   , y[indicesToKeep]
                   , c=color
                   , marker=shape
                   , s=10)
    ax.legend(['coupled', 'non-coupled'])
    ax.grid()
    plt.xscale('log')
    plt.show()

# # TESTing the plot
# print(dbdt.shape)
# print(lbl.shape)
# dbdt = np.array([[1, 2], [1, 2], [1, 2],[1, 2],[1, 2],[1, 2]])
# print(dbdt.shape)
# lbl= np.array([1, 0, 1, 1, 1, 1])
# print(lbl.shape)
# timestamp = np.array([[1], [2], [3], [4], [5], [6]])

# df2 = pd.DataFrame({'a':[1, 2, 3, 4, 5, 6],
#                    'b':[4, 5, 6, 7, 8, 9],
#                    'c':[7, 8, 9, 10, 11, 12]})

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
