import numpy as np
from itertools import combinations
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pkg_resources
import os

class DEMSRankerMixin(object):
    def demsrank(self, input_df, lead_word_weight=1, verb_spec_weight=1):
        lead_path = pkg_resources.resource_filename("sumpy", 
                        os.path.join("data", "lead_words.txt"))
        self._leadwords = []
        with open(lead_path, u"r") as f:
            text = f.readlines()
            for line in text:
                line_split = line.split()
                self._leadwords.append(line_split[0])
        verb_path = pkg_resources.resource_filename("sumpy", 
                        os.path.join("data", "verb_specificity.txt"))
        self._verbspec = {}
        with open(verb_path, u"r") as f:
            text = f.readlines()
            for line in text:
                line_split = line.split()
                self._verbspec[line_split[0]] = line_split[1]

        lead_score = np.zeros(len(input_df.index))
        verb_score = np.zeros(len(input_df.index))
        for i in range(0, len(input_df.index)):
            lead_score[i] = self._get_lead_values(input_df['text'][i])
            verb_score[i] = self._get_verb_specificity(input_df['text'][i])
        lead_score = lead_score / np.amax(lead_score)
        verb_score = verb_score / np.amax(verb_score)
        input_df[u"rank:demsrank"] = lead_score + verb_score

    def _get_lead_values(self, sent):
        word_count = 0
        lead_word_count = 0
        for token in sent:
            word_count = word_count + 1
            if token in self._leadwords:
                lead_word_count = lead_word_count + 1
        return float(lead_word_count) / word_count

    def _get_verb_specificity(self, sent):
        max_val = 0
        for token in sent:
            if token in self._verbspec.keys() and float(self._verbspec[token]) > max_val:
                max_val = float(self._verbspec[token])
        return max_val

class LedeRankerMixin(object):
    def rank_by_lede(self, input_df):
        input_df[u"rank:lede"] = 0
        input_df.loc[input_df[u"doc position"] == 1, u"rank:lede"] = 1

class TextRankMixin(object):
    def textrank(self, input_df, directed=u"undirected", d=0.85, 
                 max_iters=20, tol=.0001):
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

    def compute_kernel(self, word_sets, directed=u"undirected"):
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

class LexRankMixin(object):
    def lexrank(self, input_df, tfidf_mat, d=.85, max_iters=20, tol=.0001):
        max_sents = len(input_df)
        K = cosine_similarity(tfidf_mat)        
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
            print "lexrank failed to converged after {} iters".format(
                max_iters)
        input_df["rank:lexrank"] = r


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


