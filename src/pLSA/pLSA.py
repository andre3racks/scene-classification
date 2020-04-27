from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
import numpy as np

def create_NMF(HsofWs, num_topics):

    # create model
    model = NMF(n_components=num_topics, max_iter=300, solver='mu', beta_loss='kullback-leibler')

    # fit and transform histogram of words
    document_topic_matrix_train = model.fit_transform(HsofWs['train'])
    document_topic_matrix_test = model.transform(HsofWs['test'])

    # normalize !!!!
    norm_train = normalize_matrix(document_topic_matrix_train)
    norm_test = normalize_matrix(document_topic_matrix_test)

    return norm_train, norm_test

def normalize_matrix(mtx):
    
    mtx = np.asarray(mtx)
    normalized = []

    for z_array in mtx:
        array_sum = 0
        for element in z_array:
            array_sum += element
        
        normalized.append(z_array/array_sum)

    return normalized