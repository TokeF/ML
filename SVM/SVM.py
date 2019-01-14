import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt  
#read data
bankdata = pd.read_csv("bill_authentication.csv")  
#seperate data from labels
X = bankdata.drop('Class', axis=1)  
y = bankdata['Class']
#divide into training and test set using sklearn
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  
print(np.shape(X_train))
print(np.shape(X_test))
#import and train SVC (classifier) 
from sklearn.svm import SVC  
svclassifier = SVC(kernel='rbf', gamma='auto')  
svclassifier.fit(X_train, y_train)
#predict labels for test data
y_pred = svclassifier.predict(X_test)  
#evaluate results using built in report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test,y_pred))  
print(classification_report(y_test,y_pred))
print(y_test)
