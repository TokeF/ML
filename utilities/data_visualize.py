import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import utilities.generate_random_color as randc

def plotDat(timestamp : np.ndarray, dbdt : np.ndarray, lbl : np.ndarray):

    #find index of label change from 0 to 1 or 1 to 0. Ved at finde indeks for en inbyrdes non-zero differens
    zero_crossings = np.where(np.diff(np.transpose(lbl).tolist()))[0]
    if len(zero_crossings) == 0: print('no label change'), exit() #plot does not support no label change
    #genereate colors for plotting lines, so each gate is same color
    fig, ax = plt.subplots()
    ax.set_prop_cycle('color', plt.cm.Spectral(np.linspace(0,1,dbdt.shape[1])))
    
    #PLOT: the zerocross index, is the index before the change
    for i in range(len(zero_crossings) + 1): #always one for segment than zero changes
        #indice for first segment
        if i == 0:
            startIdx = 0
            endIdx = zero_crossings[i] + 1 + 1
            coupled = lbl[zero_crossings[i]]
        #indice for middle segment
        if 0 < i < len(zero_crossings):
            startIdx = zero_crossings[i - 1] + 1
            endIdx = zero_crossings[i] + 1 + 1
            coupled = lbl[zero_crossings[i]]
        #indice for last segment
        if i == len(zero_crossings):
            startIdx = zero_crossings[i - 1] + 1
            endIdx = None
            coupled = not lbl[zero_crossings[i - 1]]
        #plot in colour or black if inuse or not
        l = 0.5
        m = 8
        if coupled:
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
