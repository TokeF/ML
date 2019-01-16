import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '../utilities')

#load data
from data_reader import load_data
fname = "../data/20171121CH2_export.xyz"
df = load_data(fname)
rng = range(10000)
X = df.loc[rng,'DBDT_1':'DBDT_20'].values
lbl = df.loc[rng,'DBDT_INUSE_14'].values
#divide into training and test set using sklearn
from sklearn.model_selection import train_test_split  
X_train, X_test, lbl_train, lbl_test = train_test_split(X, lbl, test_size = 0.20)  
print(np.shape(X_train))
print(np.shape(X_test))
#import and train SVC (classifier) 
from sklearn.svm import SVC  
svclassifier = SVC(kernel='linear', gamma='auto')  
svclassifier.fit(X_train, lbl_train)
#predict labels for test data
lbl_pred = svclassifier.predict(X_test)  
#evaluate results using built in report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(lbl_test,lbl_pred))  
print(classification_report(lbl_test,lbl_pred))
print(lbl_test)
print(lbl_pred)