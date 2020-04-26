from visualVocab import kmeans
from pLSA import pLSA
import cv2
import numpy as np
from utils import data_loader
from classification import svm
import pickle
import os
import pprint as pp
import utils.plot as plot
import time

def main(data, NVM, list_k):

    print("reading in data...")
    # x examples are sift descriptors for images
    # y labels are scene label strings
    print("starting descriptor extraction...")
    data = data_loader.directory_walk("./dataset/train/NATURAL")

    assert len(data['X_train']) == len(data['Y_train']), "len(X_train): {}; len(Y_train): {}".format(len(data['X_train']), len(data['Y_train']))
    assert len(data['X_test']) == len(data['Y_test']), "len(X_test): {}; len(Y_test): {}".format(len(data['X_test']), len(data['Y_test']))
    
    # data['X_train'] = data['X_train'][:int(len(data['X_train'])/2)]

    num_topics_knn = 32

    test_acc, train_acc = [], []

    print("descriptor extraction looks good !!")

    for num_visual_words in NVM:
        print("obtaining histogram of words for examples...")
        # obtain histogram of words for each image in test and training set
        HsofWs = None
        model_name = "cluster_models/cluster_stride10_NAT_" + str(num_visual_words)

        if os.path.isfile(model_name):
            print("opening pre-trained cluster model for k={}.".format(num_visual_words))
            model = pickle.load(open(model_name, 'rb'))
            HsofWs = kmeans.hists_of_words(data, num_visual_words, model=model)

        else:
            HsofWs = kmeans.hists_of_words(data, num_visual_words, max_iter=300, modelname=model_name)

        assert len(HsofWs['train']) == len(data['X_train']), "HsofWs['train'] length: {} processed_data['x_train'] length: {}".format(len(HsofWs['train']), len(data['X_train']))
        assert len(HsofWs['test']) == len(data['X_test'])

        data['Z_train'] = pLSA.create_NMF(HsofWs['train'], num_topics_knn)
        data['Z_test'] = pLSA.create_NMF(HsofWs['test'], num_topics_knn)
                    
        # train_performance, test_performance = svm.kernel_SVM(data, C=10, kernel='rbf')
        for k in list_k:

            tr_knn, ts_knn = svm.KNN(data, k)
            test_acc.append(ts_knn['accuracy'])
            train_acc.append(tr_knn['accuracy'])
    
        print(ts_knn)

        # plot.plot_3D("Performance Varying V & Z", "V (# visual words)", "Z (# topics)", "Accuracy", list_num_words, list_num_topics, train_acc, test_acc)
        # plot.plot_2D("Performance Varying K (# neighbours)", "K", "Accuracy", test_acc, train_acc, list_k)
# debug purposes
if __name__ == "__main__":
    data = {}
    # data['X_train'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE), cv2.imread("output1.jpg", cv2.IMREAD_GRAYSCALE)]
    # data['X_test'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE)]
    main(data, [41], [10])