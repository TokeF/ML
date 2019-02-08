import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import utilities.PC_anal
import utilities.data_visualize as dvis
import sys
sys.path.insert(0, '../utilities')

def mvg_avg(gate : pd.DataFrame, wsize : int) :
    shift = - int(np.floor(wsize/2))
    window = gate.DBDT_Ch2GT8.shift(shift).rolling(wsize).mean()
    gate = pd.concat([window, gate.DBDT_Ch2GT8], axis =1)
    print(gate)
    return gate.values


def row_ratio(timestamp : np.ndarray, dbdt : np.ndarray) -> np.ndarray:
	#compute ratio between neighbour gates
	#allocate space for ratio vbectors/matrix
	ratio = np.zeros((dbdt.shape[0], dbdt.shape[1] - 1))
	for i in range(dbdt.shape[1] - 1):
		ratio[:, i] = dbdt[:,i+1] / dbdt[:,i]
	return ratio

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

    