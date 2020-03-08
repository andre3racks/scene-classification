import numpy as np
from sklearn.cluster import KMeans
import sift
import cv2

# generates 'visual words' from the clustering of descriptors
# input is images
# returns bag of words for images after fitting for the train images
def bag_of_words(data, k, max_iter=300, alg='auto'):
    
    # use sift file to get descriptors from train and test images
    train_kp, train_descriptors = sift.build_sift_descriptors(data['X_train'])
    test_kp, test_descriptors = sift.build_sift_descriptors(data['X_test'])

    model = KMeans(n_clusters=k, max_iter=max_iter, algorithm=alg)
    # fit model to descriptor data from training examples
    training_cluster_ass = model.fit_predict(train_descriptors)
    # predict X_test desciptors for bag of words
    testing_cluster_ass = model.predict(test_descriptors)

    training_cluster_ass = np.array(training_cluster_ass)

    print(training_cluster_ass.shape)

    bags = {}
    bags['train'] = training_cluster_ass
    bags['test'] = testing_cluster_ass

    return bags


# debug purposes
if __name__ == "__main__":

    data = {}
    data['X_train'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE)]
    data['X_test'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE)]
    print(bag_of_words(data, 50))
