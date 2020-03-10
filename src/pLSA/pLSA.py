from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
import numpy as np

def create_NMF(ex_histograms, num_topics):

    # num components should be num topics, currently not for debugging purposes
    model = NMF(n_components=1, init='nndsvda')
    # model.fit(ex_histograms)
    # print(model.components_)


    document_topic_matrix = model.fit_transform(ex_histograms)
    # shape is (num examples, num topics)
    # find max of row to give topic assignment
    return document_topic_matrix