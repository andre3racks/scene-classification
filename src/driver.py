from visualVocab import kmeans
from pLSA import pLSA
import cv2
import numpy as np
from utils import data_loader
from classification import svm


def main(data):

    print("reading in data...")
    # x examples are sift descriptors for images
    # y labels are scene label strings
    print("starting descriptor extraction...")
    data = data_loader.directory_walk("./dataset/train")

    assert len(data['X_train']) == len(data['Y_train']), "len(X_train): {}; len(Y_train): {}".format(len(data['X_train']), len(data['Y_train']))
    assert len(data['X_test']) == len(data['Y_test']), "len(X_test): {}; len(Y_test): {}".format(len(data['X_test']), len(data['Y_test']))
    
    print("descriptor extraction looks good !!")


    print("obtaining histogram of words for examples...")
    # obtain histogram of words for each image in test and training set
    # obtain data after deletions of any sift failures
    HsofWs = kmeans.hists_of_words(data, 10, max_iter=30)

    assert len(HsofWs['train']) == len(data['X_train']), "HsofWs['train'] length: {} processed_data['x_train'] length: {}".format(len(HsofWs['train']), len(data['X_train']))
    assert len(HsofWs['test']) == len(data['X_test'])

    print("factoring pLSA matrix...")
    # NMF model for word / topic relationship
    num_topics = 10
    data['Z_train'] = pLSA.create_NMF(HsofWs['train'], num_topics)
    data['Z_test'] = pLSA.create_NMF(HsofWs['test'], num_topics)
    
    # doc topic matrix holds Z arrays for each example
    # Z arrays show an examples relationship to the topics
    print("training SVM...")
    # pass Z arrays and labels to SVM for training and testing
    scores = svm.SVM(data, 5)

    print(scores)

    
    return 

# debug purposes
if __name__ == "__main__":
    data = {}
    # data['X_train'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE), cv2.imread("output1.jpg", cv2.IMREAD_GRAYSCALE)]
    # data['X_test'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE)]
    main(data)