from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF
import numpy as np

def create_NMF(ex_histograms, num_topics):

    # num components should be num topics, currently not for debugging purposes
    model = NMF(n_components=1, init='nndsvda')

    document_topic_matrix = model.fit_transform(ex_histograms)

    # normalize !!!!

    return document_topic_matrix