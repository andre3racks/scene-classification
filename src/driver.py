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

def main(data, NVM, k_list):

    print("reading in data...")
    # x examples are sift descriptors for images
    # y labels are scene label strings
    print("starting descriptor extraction...")
    data = data_loader.directory_walk("./dataset/train/NATURAL")

    assert len(data['X_train']) == len(data['Y_train']), "len(X_train): {}; len(Y_train): {}".format(len(data['X_train']), len(data['Y_train']))
    assert len(data['X_test']) == len(data['Y_test']), "len(X_test): {}; len(Y_test): {}".format(len(data['X_test']), len(data['Y_test']))
    
    # data['X_train'] = data['X_train'][:int(len(data['X_train'])/2)]

    num_topics_knn = 29

    test_acc, train_acc = [], []
    plot_k, plot_t_list = [], []
    print("descriptor extraction looks good !!")

    for num_visual_words in NVM:
        print("obtaining histogram of words for examples...")
        # obtain histogram of words for each image in test and training set
        HsofWs = None
    
        model_name = "cluster_models/cluster_stride10_nat_" + str(num_visual_words)
        # load the kmeans model if found
        if os.path.isfile(model_name):
            print("opening pre-trained cluster model for k={}.".format(num_visual_words))
            model = pickle.load(open(model_name, 'rb'))
            # retrieve histograms of visual word frequencies
            HsofWs = kmeans.hists_of_words(data, num_visual_words, model=model)

        else:
            # retrieve histograms of visual word frequencies
            HsofWs = kmeans.hists_of_words(data, num_visual_words, max_iter=300, modelname=model_name)

        # check len constraints
        assert len(HsofWs['train']) == len(data['X_train']), "HsofWs['train'] length: {} processed_data['x_train'] length: {}".format(len(HsofWs['train']), len(data['X_train']))
        assert len(HsofWs['test']) == len(data['X_test'])

        for k in k_list:
            # obtain doc / topic matrices
            data['Z_train'], data['Z_test'] = pLSA.create_NMF(HsofWs, num_topics_knn)
                        
            # train and test KNN classifier
            tr_knn, ts_knn = svm.KNN(data, k)
            test_acc.append(ts_knn['accuracy'])
            train_acc.append(tr_knn['accuracy'])

            # plotting functions
            plot_k.append(k)
            # plot_t_list.append(topics)
    
        print(ts_knn)

    ind = np.argpartition(test_acc, -1)[-1:]
    print("best performers")
    for i in ind:
        print("K: {}, acc: {}".format(plot_k[i], test_acc[i]))

    # plot.plot_3D("Performance Varying V & Z", "V (# visual words)", "Z (# topics)", "Accuracy", plot_NVM, plot_t_list, train_acc, test_acc)
    # plot.plot_2D("Performance Varying K (# neighbours)", "K", "Accuracy", test_acc, train_acc, plot_k)

# debug purposes

if __name__ == "__main__":
    data = {}
    # data['X_train'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE), cv2.imread("output1.jpg", cv2.IMREAD_GRAYSCALE)]
    # data['X_test'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE)]
    main(data, [83], [26])
