from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import NMF


def create_NMF(ex_histograms, num_topics):

    # num components should be num topics, currently not for debugging purposes
    model = NMF(n_components=2, init='nndsvda')
    model.fit(ex_histograms)
    # print(model.components_)
    

    return model