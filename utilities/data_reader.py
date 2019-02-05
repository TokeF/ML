import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def load_data(fileName : str): 
    #read .xyz datafile into pandas data frame
    dataFrame = pd.read_csv(fileName, sep='\s+',header=[18])
    #correct header names
    #first column is a \ due to data file. Thus labels are shifted one back. Column 179 is empty and thus deleted
    header = dataFrame.columns.drop('/')
    dataFrame.drop("DBDT_INUSE_Ch2GT33", axis=1, inplace=True)
    dataFrame.columns = header
    #assert that there are no NAN or dummy values
    assert not dataFrame.isnull().values.any(), "NAN value in data: " + str(dataFrame.isnull().sum().sum())
    a = dataFrame == 9999
    assert not a.any().any(), 'Dummy values are present'
    return dataFrame

def load_data2(fileName : str, gF : int, gT : int, inuse : int):
    dataFrame = load_data(fileName)
    gFrom = 'DBDT_Ch2GT' + str(gF)
    gTo = 'DBDT_Ch2GT' + str(gT)
    lblGate = 'DBDT_INUSE_Ch2GT' + str(inuse)
    dbdt = dataFrame.loc[:,gFrom:gTo].values
    lbl = dataFrame.loc[:,lblGate:lblGate].values
    return dataFrame, dbdt, lbl, dataFrame.loc[:, 'TIMESTAMP'].values