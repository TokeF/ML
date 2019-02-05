import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import PC_anal
import data_visualize as dvis
import sys
sys.path.insert(0, '../utilities')

#load data
from data_reader import load_data2
fname = "../data/20171101_RAW_export.xyz"
_ , dbdt, lbl, timestamp = load_data2(fname, 8, 20, 14)

#compute ratio between neighbour gates
#allocate space for ratio vbectors/matrix
r = np.zeros((dbdt.shape[0], dbdt.shape[1] - 1))
for i in range(dbdt.shape[1] - 1):
    r[:, i] = dbdt[:,i+1] / dbdt[:,i]

dvis.plotDat(timestamp , r, lbl)
plt.xlim([timestamp[0], timestamp[700]])
plt.yscale('log')
lbl = lbl[:,0]
PC_anal.PCA_custom(r, lbl, 'ratio')
plt.show()
input()

#Compute difference between rows, UNUSED!
def row_diff(dbdt : np.ndarray):
    #for each column, compute the distance to all columns on the right
    n = dbdt.shape[1] - 1
    triangularNum =  int(n * (n + 1)/2) 
    a = np.zeros((dbdt.shape[0], triangularNum))
    j = 0
    for i in range(n):
        for h in  range(i,n):
            a[:, j] = dbdt[:,i+1] - dbdt[:,i]
            j += 1
    return a

    