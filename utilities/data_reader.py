import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def load_data(fileName : str): 
    #read .xyz datafile into pandas data frame
    dataFrame = pd.read_csv(fileName, sep='\s+',header=[18])
    assert dataFrame.columns[1] == 'TIMESTAMP', "Header is wrong. Location 1 yield: " + str(dataFrame.columns[1])
    #correct header names
    #first column is a \ due to data file. Thus labels are shifted one back. Column 179 is empty and thus deleted
    header = dataFrame.columns.drop('/')
    dataFrame.drop("DBDT_INUSE_Ch2GT33", axis=1, inplace=True)
    dataFrame.columns = header
    #assert that there are no NAN or dummy values
    assert not dataFrame.isnull().values.any(), "NAN value in data: " + str(dataFrame.isnull().sum().sum())
    return dataFrame

def load_data2(fileName : str, gF : int, gT : int):
    dataFrame = load_data(fileName)
    gFrom = 'DBDT_Ch2GT' + str(gF)
    gTo = 'DBDT_Ch2GT' + str(gT)
    lblFrom = 'DBDT_INUSE_Ch2GT' + str(gF)
    lblTo = 'DBDT_INUSE_Ch2GT' + str(gT)
    dbdt = dataFrame.loc[:,gFrom:gTo].values
    lbl = dataFrame.loc[:,lblFrom:lblTo].values
    dumdbdt = dbdt == 99999
    dumlbl = lbl == 99999
    assert not dumdbdt.any().any(), 'Dummy values in DBDT present: ' + str(dumdbdt.any().sum().sum())
    assert not dumlbl.any().any(), 'Dummy values in label present: ' + str(dumlbl.any().sum().sum())
    assert (lbl.T == lbl[:,0]).any(), 'Labels are not equal'
    lbl = lbl[:,0]
    return dataFrame, dbdt, lbl, dataFrame.loc[:, 'TIMESTAMP'].values