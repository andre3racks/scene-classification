import numpy as np
from sklearn.cluster import KMeans
import visualVocab.sift as sift
import cv2

# generates 'visual words' from the clustering of descriptors
# input is images
# returns bag of words for images after fitting for the train images
def hists_of_words(data, k, max_iter=300, alg='auto', model=None):
    # flatten descriptors for k means input
    train_descriptors = np.concatenate(data['X_train'], axis=0)
    test_descriptors = np.concatenate(data['X_test'], axis=0)

    if model is None:
        print("clustering train descriptors...")
        model = KMeans(n_clusters=k, max_iter=max_iter, algorithm=alg)
    else:
        TODO = None
        # load previously trained model
        
    # fit model to descriptor data from training examples
    training_cluster_ass = model.fit_predict(train_descriptors)
    # predict X_test desciptors for bag of words
    print("predicting test descriptors...")
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
        
        for elements in bag:
            hist[elements] += 1

        # normalize hist
        hist = hist / len(bag)

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
