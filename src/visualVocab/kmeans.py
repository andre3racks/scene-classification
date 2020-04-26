import numpy as np
from sklearn.cluster import KMeans
import visualVocab.sift as sift
import cv2
import pickle
import time

# generates 'visual words' from the clustering of descriptors
# input is images
# returns bag of words for images after fitting for the train images
def hists_of_words(data, k, max_iter=300, alg='auto', model=None, modelname=None):
    
    # flatten descriptors for k means input
    train_descriptors = np.concatenate(data['X_train'], axis=0)
    train_descriptors = np.asarray(train_descriptors)
    print("train descriptor shape:  {}".format(train_descriptors.shape))
    test_descriptors = np.concatenate(data['X_test'], axis=0)

    if model is None:
        start = time.time()
        print("clustering train descriptors...")
        model = KMeans(n_clusters=k, max_iter=max_iter, algorithm=alg)
        
        assert modelname is not None, "modelname is none"
        model_name = modelname
        training_cluster_ass = model.fit_predict(train_descriptors)
        print("Saving the model as {}".format(model_name))
        pickle.dump(model, open(model_name, 'wb'))
        print("clustering took: {}".format((time.time() - start)/60))
        print("finished clustering. creating test bag of words...")
        testing_cluster_ass = model.predict(test_descriptors)

    else:
        # load previously trained model
        training_cluster_ass = model.predict(train_descriptors)
        testing_cluster_ass = model.predict(test_descriptors)
            
    train_bags_size = []
    for bag in data['X_train']:
        train_bags_size.append(len(bag))

    test_bag_size = []
    for bag in data['X_test']:
        test_bag_size.append(len(bag))

    bags = {}
    # bags['train'] = training_cluster_ass
    bags['train'] = bag_to_histogram(k, train_bags_size, training_cluster_ass)
    bags['test'] = bag_to_histogram(k, test_bag_size, testing_cluster_ass)

    return bags

# takes in bag sizes and list of bags of words to create histograms for each example
def bag_to_histogram(k, bag_sizes, kmeans_output):
    histograms = []

    last = 0

    for hist_size in bag_sizes:
        hist = np.zeros(k)
        bag = kmeans_output[last:hist_size+last]
        # bags are all the same???
        # print("bag: {}".format(bag))
        for elements in bag:
            hist[elements] += 1

        # normalize hist
        hist = hist / len(bag)

        # print("hist: {}".format(hist))

        histograms.append(hist)
        # print(hist)

        last += hist_size

    return histograms

# debug purposes
if __name__ == "__main__":
    data = {}
    data['X_train'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE), cv2.imread("output1.jpg", cv2.IMREAD_GRAYSCALE)]
    data['X_test'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE)]
    print(hists_of_words(data, 3))
