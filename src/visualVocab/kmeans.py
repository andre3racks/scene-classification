import numpy as np
from sklearn.cluster import KMeans
import visualVocab.sift as sift
import cv2

# generates 'visual words' from the clustering of descriptors
# input is images
# returns bag of words for images after fitting for the train images
def hists_of_words(data, k, max_iter=300, alg='auto'):

    # use sift file to get descriptors from train and test images
    train_bags_size, train_descriptors = sift.build_sift_descriptors(data['X_train'])
    test_bag_size, test_descriptors = sift.build_sift_descriptors(data['X_test'])

    # print("train bag sizes: {}".format(train_bags_size))
    model = KMeans(n_clusters=k, max_iter=max_iter, algorithm=alg)
    # fit model to descriptor data from training examples
    training_cluster_ass = model.fit_predict(train_descriptors)
    # predict X_test desciptors for bag of words
    testing_cluster_ass = model.predict(test_descriptors)

    training_cluster_ass = np.array(training_cluster_ass)

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
        
        for elements in bag:
            hist[elements] += 1

        histograms.append(hist)
        # print(hist)

        last = hist_size

    return histograms

# debug purposes
if __name__ == "__main__":
    data = {}
    data['X_train'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE), cv2.imread("output1.jpg", cv2.IMREAD_GRAYSCALE)]
    data['X_test'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE)]
    print(hists_of_words(data, 3))
