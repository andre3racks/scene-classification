import numpy as np
from sklearn import svm
from sklearn.metrics import confusion_matrix

# can pass in pre-trained models
def SVM(data, C, model=None):
    
    if model is None:
        model = svm.SVC(C = C)
        model.fit(data['Z_train'], data['Y_train'])
    
    return generate_results(data['Z_test'], data['Y_test'], model)


def kernel_SVM(data, C, gamma, model=None):

    if model is None:
        model = svm.SVC(C=C, kernel='rbf', gamma=gamma)
        model.fit(data['Z_train'], data['Y_train'])

    return generate_results(data['Z_test'], data['Y_test'], model)

# taken from a3
def generate_results(X, y, model):
    results = {"accuracy" : 0,
               "recall" : 0,
               "precision" : 0,
               "avg_recall" : 0,
               "avg_precision" : 0,
               "fscore" : 0}

    pred = model.predict(X)

    print("Creating confusion matrix and calculating evaluation metrics")
    #Calculate the confusion matrix, and normalize it between 0-1
    cm = confusion_matrix(y,pred).astype(np.float32)
    # epsilon for non zero entries
    EPS=1e-6

    #From the confusion matrix, calculate precision/recall/f1-measure
    results['recall'] = np.diag(cm) / (np.sum(cm, axis=1) + EPS)
    results['avg_recall'] = np.mean(results['recall'])

    results['precision'] = np.diag(cm) / (np.sum(cm, axis=0) + EPS)
    results['avg_precision'] = np.mean(results['precision'])

    results["fscore"] = 2 * ( results['avg_precision'] * results['avg_recall'] ) / (results['avg_precision'] + results['avg_recall'] + EPS)

    results["accuracy"] = np.mean(pred == y)

    return results