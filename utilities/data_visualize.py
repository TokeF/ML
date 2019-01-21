import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import generate_random_color as randc

fname = "../data/20171123CH2_export.xyz"
from data_reader import load_data
df = load_data(fname)
# working on visualization tool
dbdt = df.loc[:,'DBDT_5':'DBDT_15'].values
lbl = df.loc[:,'DBDT_INUSE_10':'DBDT_INUSE_10'].values
#check if all lbls are equal by comparing first row with the rest (therefore the transpose)
lbl2 = np.transpose(lbl)
print("All labels are identical: " + str((lbl2 == lbl2[0,:]).all()))

def plotDat(dbdt : np.ndarray, lbl : np.ndarray):
    #find index of label change from 0 to 1 or 1 to 0. Ved at finde indeks for en inbyrdes non-zero differens
    zero_crossings = np.where(np.diff(np.transpose(lbl).tolist()))[1]

    to = len(lbl)
    x = range(to)

    # #generate n colors where n is number fo gates //should be made better (just ask for n colors)
    # colors = []
    # for i in range(dbdt.shape[1]):
    #     colors.append(randc.generate_new_color(colors,pastel_factor = 0.9))

    #genereate colors for plotting lines, so each gate is same color
    fig, ax = plt.subplots()
    ax.set_prop_cycle('color', plt.cm.Spectral(np.linspace(0,1,dbdt.shape[1])))
    # print(colors)
    # exit()
    #plot first segment
    firstidx = zero_crossings[0] + 1
    if lbl[zero_crossings[0]] == 0:
        plt.plot(x[0:firstidx], dbdt[: , 0:firstidx])
    else:
        plt.plot(x[0:firstidx], dbdt[0:firstidx, :], color = 'black')
    #plot middle segments
    for i,idx in enumerate(zero_crossings):
        if i + 1 != len(zero_crossings):
            startIdx = idx + 1
            endIdx = zero_crossings[i + 1] + 1
            if lbl[idx + 1] == 0: 
                plt.plot(x[startIdx : endIdx], dbdt[startIdx : endIdx, :])
            else:
                plt.plot(x[startIdx : endIdx], dbdt[startIdx : endIdx, :], color = 'black')
        #plot very last segment
        else:
            if lbl[idx + 1] == 0:
                plt.plot(x[idx + 1 : ], dbdt[idx + 1 : , :])
            else:
                plt.plot(x[idx + 1 : ], dbdt[idx + 1 : , :], color = 'black')
    # plt.xlim(200,700)
    # plt.show()

plotDat(dbdt, lbl)
differenceDBDT = np.diff(dbdt, axis = 1)
plotDat(differenceDBDT, lbl)
plt.show()
exit()


#Normalize
a = (a - np.amin(a)) / (np.amax(a) - np.amin(a))

# colors = ['red','green','blue','purple']
# plt.plot(x, lbl[0:to])
for i in range(20):
    plt.plot(x,a[0:to,i])
    # plt.scatter(x,a[0:to,i], c=lbl[0:to], cmap= matplotlib.colors.ListedColormap(colors))
# plt.ylim(np.amin(a),np.amax(a))
plt.show()