import numpy as np
from sklearn import svm

def SVM(data, C):
    
    model = svm.SVC(C = C)
    model.fit(data['Z_train'], data['Y_train'])

    scores = {}
    scores["test"] = model.score(data['Z_test'], data['Y_test'])
    scores["train"] = model.score(data['Z_train'], data['Y_train'])

    return scores

def kernel_SVM(data, C, gamma):

    model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
    model.fit(data['Z_train'], data['Y_train'])

    scores = {}
    scores["test"] = model.score(data['Z_test'], data['Y_test'])
    scores["train"] = model.score(data['Z_train'], data['Y_train'])

    return scores