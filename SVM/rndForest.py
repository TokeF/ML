import pandas as pd  
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification

#load data
import sys
sys.path.insert(0, '../utilities')
from data_reader import load_data
fname = "../data/20171101_RAW_export.xyz"
df = load_data(fname)
dbdt = df.loc[:,'DBDT_Ch2GT2':'DBDT_Ch2GT25'].values
lbl = df.loc[:,'DBDT_INUSE_Ch2GT14'].values

# ## DELETE this it uis for wuick attempt at using ratio
# dbdt2 = dbdt.values
# r = np.zeros((dbdt2.shape[0], dbdt2.shape[1] - 1))
# for i in range(dbdt2.shape[1] - 1):
#     r[:, i] = dbdt2[:,i+1] / dbdt2[:,i]
# ## DELETE
# X = np.concatenate((dbdt, r), axis = 1)
X = dbdt
X_train, X_test, lbl_train, lbl_test = train_test_split(X, lbl, test_size = 0.20)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()  
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# #apply random undersampling
# from imblearn.under_sampling import RandomUnderSampler
# from imblearn.over_sampling import RandomOverSampler
# rus = RandomOverSampler(return_indices=True)
# X_train, lbl_train, id_rus = rus.fit_sample(X_train, lbl_train)

# Make classification
classifier = RandomForestClassifier(n_estimators=100)  
classifier.fit(X_train, lbl_train)  
lbl_scor = classifier.predict_proba(X_test)
lbl_pred = lbl_scor[:][:,1].round()
misclassified = np.where(lbl_test != lbl_pred)
plt.hist(lbl_scor[misclassified[0],:][:][:,1])
b = lbl_scor[:][:,1]
c = b[b >= 0.85]
b = b[b < 0.85]
print(b.shape)
print(c.shape)

print(misclassified)
exit()
#metrics
from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(lbl_test,lbl_pred))  
print(classification_report(lbl_test,lbl_pred))
print(lbl_test)
print(lbl_pred)
plt.show()
# scores = cross_val_score(classifier, X, lbl, cv=5)
# print(scores.mean())

# ## Isolation forest
# uncoupled = dbdt.loc[df['DBDT_INUSE_Ch2GT14'] == 1].values
# coupled = dbdt.loc[df['DBDT_INUSE_Ch2GT14'] == 0].values
# print(coupled.shape[0] / uncoupled.shape[0])
# X_trainU, X_testU = train_test_split(uncoupled, test_size = 0.20)

# # Feature Scaling
# sc2 = StandardScaler()  
# X_trainU = sc2.fit_transform(X_trainU)  
# X_testU = sc2.transform(X_testU)

# sc3 = StandardScaler()  
# y_c = sc2.transform(coupled)

# from sklearn.ensemble import IsolationForest
# clf = IsolationForest(n_estimators = 1000, contamination = 0.13)
# clf.fit(X_train)
# # predictions
# X_pred_test = clf.predict(X_test)
# y_pred_outliers = clf.predict(y_c)
# print("Accuracy test:", list(X_pred_test).count(1)/X_pred_test.shape[0])
# print("Accuracy outlie: ", list(y_pred_outliers).count(-1)/y_pred_outliers.shape[0])
