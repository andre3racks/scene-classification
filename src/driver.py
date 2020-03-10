from visualVocab import kmeans
from pLSA import pLSA
import cv2
import numpy as np


def main(data):

    # obtain histogram of words for each image in test and training set
    HsofWs = kmeans.hists_of_words(data, 200)
    # NMF model for word / topic relationship
    num_topics = 10
    model = pLSA.create_NMF(HsofWs['train'], num_topics)

    word_dict = {}
    n_top_words = 1
    
    # should print top word for each topic when we run on all data
    # for i in range(num_topics):
    #     words_ids = model.components_[i].argsort()[:-n_top_words - 1:-1]
    #     words = [key for key in words_ids]
    #     word_dict['Topic # ' + '{:02d}'.format(i+1)] = words

    # print(word_dict)

    return 

# debug purposes
if __name__ == "__main__":
    data = {}
    data['X_train'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE), cv2.imread("output1.jpg", cv2.IMREAD_GRAYSCALE)]
    data['X_test'] = [cv2.imread("output2.jpg", cv2.IMREAD_GRAYSCALE)]
    main(data)