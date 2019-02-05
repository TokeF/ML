import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import generate_random_color as randc

def plotDat(timestamp : np.ndarray, dbdt : np.ndarray, lbl : np.ndarray):

    #find index of label change from 0 to 1 or 1 to 0. Ved at finde indeks for en inbyrdes non-zero differens
    zero_crossings = np.where(np.diff(np.transpose(lbl).tolist()))[1]

    #genereate colors for plotting lines, so each gate is same color
    fig, ax = plt.subplots()
    ax.set_prop_cycle('color', plt.cm.Spectral(np.linspace(0,1,dbdt.shape[1])))
    
    for i,idx in enumerate(zero_crossings):
        #indice for first segment
        if i == 0:
            startIdx = 0
            endIdx = idx + 1
            idx -= 1
        #indice for middle segment
        if 0 < i < len(zero_crossings) - 1:
            startIdx = idx + 1
            endIdx = zero_crossings[i + 1] + 1
        #indice for last segment
        if i == len(zero_crossings) - 1:
            startIdx = idx + 1
            endIdx = None
        #plot in colour or black if inuse or not
        l = 0.5
        m = 6
        if lbl[idx + 1] == 1:
            plt.plot(timestamp[startIdx : endIdx], dbdt[startIdx : endIdx, :],'.-', linewidth=l, markersize=m)
        else:
            plt.plot(timestamp[startIdx : endIdx], dbdt[startIdx : endIdx, :],'.-', color = 'black', linewidth=l, markersize=m)



def plotDat2(timestamp : np.ndarray, dbdt : np.ndarray, lbl : np.ndarray):
    #setup figure
    fig = plt.figure(figsize = (8,8))
    ax = fig.add_subplot(111)
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_title('jkas: ', fontsize = 20)

    #Set target ie. inuse or not, color and shape for data
    targets = [0, 1]
    colors = ['r', 'g']
    shapes = ['o', '^']
    print(dbdt.shape)
    print(timestamp.shape)
    #iterate over a tuple (zip), scatter plot 0 and 1
    for target, color, shape in zip(targets,colors, shapes):
       indicesToKeep = lbl == target
       print(indicesToKeep[:,0].shape)
       ax.plot(timestamp[indicesToKeep[:,0]]
                ,dbdt[indicesToKeep[:,0], :]
                , c = color
                , marker = shape)
    ax.legend(['coupled', 'non-coupled'])
    ax.grid()
    plt.show()
