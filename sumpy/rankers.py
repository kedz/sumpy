import numpy as np
from itertools import combinations
from scipy.sparse import csr_matrix

class LedeRankerMixin(object):
    def rank_by_lede(self, input_df):
        input_df["rank:lede"] = 0
        input_df.loc[input_df["doc position"] == 1, "rank:lede"] = 1

class TextRankMixin(object):
    def textrank(self, input_df, d=0.85, max_iters=20, tol=.00000001):
        word_sets = [set(words) for words in input_df[u"words"].tolist()]
        max_sents = len(word_sets)
        K = self.compute_kernel(word_sets)
        M_hat = (d * K) + \
                (float(1 - d) / max_sents) * np.ones((max_sents, max_sents))
        M_hat /=  np.sum(M_hat, axis=0)
        r = np.ones((max_sents), dtype=np.float64) / max_sents

        converged = False
        for n_iter in xrange(max_iters):
            last_r = r
            r = np.dot(M_hat, r)

            if (np.abs(r - last_r) < tol).any():
                converged = True
                break

        if not converged:
            print "warning:", 
            print "textrank failed to converged after {} iters".format(
                max_iters)
        input_df["rank:textrank"] = r

    def compute_kernel(self, word_sets):
        """Compute similarity matrix K ala text rank paper. Should this be
        a ufunc???"""
        n_sents = len(word_sets)
        M = np.zeros((n_sents, n_sents), dtype=np.float64)
        for i, j in combinations(xrange(n_sents), 2):
            s_i = word_sets[i]
            s_j = word_sets[j] 
            val = len(s_i.intersection(s_j))
            val /= np.log(len(s_i) * len(s_j))
            M[i,j] = val
            M[j,i] = val
        return M

class CentroidScoreMixin(object):
    def centroid_score(self, input_df, tfidf_mat):
        centroid = tfidf_mat.sum(axis=0)
        assert centroid.shape[1] == tfidf_mat.shape[1]
        indices = tfidf_mat.indices
        indptr = tfidf_mat.indptr
        nnz = tfidf_mat.nnz
        occurence_mat = csr_matrix(
            (np.ones((nnz)), indices, indptr), shape=tfidf_mat.shape)
        input_df[u"rank:centroid_score"] = occurence_mat.dot(centroid.T)


