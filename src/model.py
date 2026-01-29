from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr

def build_vectorizer(corpus):
    vectorizer = TfidfVectorizer(analyzer='word')
    tfidf_matrix = vectorizer.fit_transform(corpus)
    return vectorizer, tfidf_matrix

def compute_similarities(query, df, vectorizer, tfidf_matrix):
    query_vec = vectorizer.transform(query).toarray()

    cosine_scores = []
    pearson_scores = []

    for i in range(tfidf_matrix.shape[0]):
        doc_vec = tfidf_matrix[i].toarray()

        cosine_scores.append(
            cosine_similarity(query_vec, doc_vec)[0][0]
        )

        pearson_scores.append(
            pearsonr(query_vec.squeeze(), doc_vec.squeeze())[0]
        )

    return cosine_scores, pearson_scores
