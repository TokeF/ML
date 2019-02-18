import pandas as pd  
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import sys
sys.path.insert(0, '../utilities')
from utilities.data_reader import load_data2
import utilities.data_visualize
import utilities.difference
def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def rnd_forest(timestamp : np.ndarray, dbdt : np.ndarray, lbl : np.ndarray, tSize : float, n_trees : int, removelow : bool = False):
    X = list(chunks(range(len(lbl)), 5))
    lbl2 = list(chunks(range(len(lbl)), 5))
    # X = range(dbdt.shape[0])
    X_train_idx, X_test_idx, lbl_train_idx, lbl_test_idx = train_test_split(X, lbl2, test_size = tSize)
    # X_test_idx = X[0:int(np.ceil(len(X)/2))]
    # lbl_test = lbl[X[0:int(np.ceil(len(X)/2))]]
    # X_train_idx = X[int(np.ceil(len(X)/2)) + 1 : ]
    # lbl_train = lbl[X[int(np.ceil(len(X)/2)) + 1 : ]]

    #For segmenting the data
    X_train_idx = [item for sublist in X_train_idx for item in sublist] #list comprehension :O
    X_test_idx = [item for sublist in X_test_idx for item in sublist]
    lbl_train_idx = [item for sublist in lbl_train_idx for item in sublist]
    lbl_test_idx = [item for sublist in lbl_test_idx for item in sublist]
    lbl_train = lbl[lbl_train_idx]
    lbl_test = lbl[lbl_test_idx]

    X_train = dbdt[X_train_idx,:]
    X_test = dbdt[X_test_idx,:]

    # Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.fit_transform(X_test)

    # apply random undersampling
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.over_sampling import RandomOverSampler
    rus = RandomOverSampler(return_indices=True)
    X_train, lbl_train, id_rus = rus.fit_sample(X_train, lbl_train)
    # X_test, lbl_test, id_rus_test = rus.fit_sample(X_test, lbl_test)

    # Make classification
    classifier = RandomForestClassifier(n_estimators=n_trees)
    classifier.fit(X_train, lbl_train)
    lbl_scor = classifier.predict_proba(X_test)

    if (removelow):
        # remove low precision
        lbl_scorA = np.asarray(lbl_scor)
        b = (lbl_scorA[:, 1] < 0.3) | (lbl_scorA[:, 1] > 0.7)
        pred = lbl_scorA[b, 1].round()
        test = lbl_test[b]
        print(len(pred) / len(lbl_test))
        #metrics
        from sklearn.metrics import classification_report, confusion_matrix
        print(confusion_matrix(test,pred))
        print(classification_report(test,pred))
        exit()

    #find misclassified samples
    lbl_pred = lbl_scor[:][:,1].round()
    misclassified = np.where(lbl_test != lbl_pred)
    corclassified = np.where(lbl_test == lbl_pred)

    #plot data and red bars where data is misclassified
    # data_visualize.plotDat(timestamp, dbdt, lbl )
    plt.yscale('log')
    for xc in misclassified[0]:
        ogmark = timestamp[X_test_idx[xc]]
        plt.axvline(x=ogmark, color = 'red')
    #plot correctly classified
    for xc in corclassified[0]:
        ogmark = timestamp[X_test_idx[xc]]
        plt.axvline(x=ogmark, color = 'blue')

    #metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print(confusion_matrix(lbl_test,lbl_pred))
    print(classification_report(lbl_test,lbl_pred))
    print("Test labels: " + str(lbl_test))
    print("predicted labels: " + str(lbl_pred))
    return lbl_scor

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
