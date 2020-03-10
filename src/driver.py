from visualVocab import kmeans
from pLSA import pLSA
import cv2
import numpy as np


def main(data):

    # obtain histogram of words for each image in test and training set
    HsofWs = kmeans.hists_of_words(data, 200)
    # NMF model for word / topic relationship
    num_topics = 10
    doc_topic_matrix = pLSA.create_NMF(HsofWs['train'], num_topics)

    # use DT matrix to find topic assignments to examples; which in essence should be their classification?

    return 

# debug purposes
if __name__ == "__main__":
    data = {}
    data['X_train'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE), cv2.imread("output1.jpg", cv2.IMREAD_GRAYSCALE)]
    data['X_test'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE)]
    main(data)