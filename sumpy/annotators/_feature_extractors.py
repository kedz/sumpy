from sumpy.annotators import (SentenceTokenizerMixin, BinaryBOWMixin, 
    TfIdfMixin, TfIdfCosineSimilarityMixin)
import numpy as np
from itertools import combinations

class LedeMixin(SentenceTokenizerMixin):

    def build(self):
        pass

    def process(self, input_df, ndarray_data):
        input_df[u"f:lede"] = 0
        for doc_id, group in input_df.groupby("doc id"):
            idx = group["sent id"].argmin()
            input_df.ix[idx, u"f:lede"]  = 1
        return input_df, ndarray_data

    def requires(self):
        return ["sent id",]

    def ndarray_requires(self):
        return []

    def returns(self):
        return ["f:lede"]

    def ndarray_returns(self):
        return []

    def name(self):
        return "LedeMixin"

class TextRankMixin(BinaryBOWMixin):

    def build(self):
        if not hasattr(self, "directed"):
            self.directed = u"undirected"
        assert self.directed in ["undirected",] #  [u"directed", "undirected"]
        # TODO actually implement directed
        
        if not hasattr(self, "d"):
            self.d = .85
        assert 0 < self.d and self.d < 1

        if not hasattr(self, "max_iters"):
            self.max_iters = 20
        assert isinstance(self.max_iters, int) and self.max_iters > 0

        if not hasattr(self, "tol"):
            self.tol = .0001
        assert 0 < self.tol
        
        def textrank(input_df, ndarray_data):
            max_sents = input_df.shape[0]
            l = input_df["words"].apply(len).tolist()
            K = self._textrank_kernel(
                l, ndarray_data["BinaryBOWMatrix"], directed=self.directed)
            M_hat = (self.d * K) + \
                    (float(1 - self.d) / max_sents) * np.ones(
                        (max_sents, max_sents))
            M_hat /=  np.sum(M_hat, axis=0)
            r = np.ones((max_sents), dtype=np.float64) / max_sents

            converged = False
            for n_iter in xrange(self.max_iters):
                last_r = r
                r = np.dot(M_hat, r)

                if (np.abs(r - last_r) < self.tol).any():
                    converged = True
                    break

            if not converged:
                print "warning:", 
                print "textrank failed to converged after {} iters".format(
                    self.max_iters)
            input_df["f:textrank"] = r
            return input_df, ndarray_data
        self._textrank = textrank

    def process(self, input_df, ndarray_data):
        return self._textrank(input_df, ndarray_data)

    def _textrank_kernel(self, l, X, directed=u"undirected"):
        """Compute similarity matrix K ala text rank paper. Should this be
        a ufunc???"""
        #X = X.todense()
        #X[X > 0] = 1
        N = X.dot(X.T)

        n_sents = X.shape[0]
        M = np.zeros((n_sents, n_sents), dtype=np.float64)
        for i, j in combinations(xrange(n_sents), 2):
            #s_i = word_sets[i]
            #s_j = word_sets[j] 
            val = N[i,j]  #len(s_i.intersection(s_j))
            val /= np.log(l[i] * l[j])
            M[i,j] = val
            M[j,i] = val
        return M

    def requires(self):
        return ["words",]
    
    def ndarray_requires(self):
        return ["BinaryBOWMatrix",]

    def returns(self):
        return ["f:textrank"]

    def ndarray_returns(self):
        return []

    def name(self):
        return "TextRankMixin"

class LexRankMixin(TfIdfCosineSimilarityMixin):

    def build(self):
        if not hasattr(self, "d"):
            self.d = .85
        assert 0 < self.d and self.d < 1

        if not hasattr(self, "max_iters"):
            self.max_iters = 20
        assert isinstance(self.max_iters, int) and self.max_iters > 0

        if not hasattr(self, "tol"):
            self.tol = .0001
        assert 0 < self.tol
        
        def lexrank(input_df, ndarray_data):
            max_sents = input_df.shape[0]
            #l = input_df["words"].apply(len).tolist()
            K = ndarray_data["TfIdfCosSimMatrix"]
            M_hat = (self.d * K) + \
                    (float(1 - self.d) / max_sents) * np.ones(
                        (max_sents, max_sents))
            M_hat /=  np.sum(M_hat, axis=0)
            r = np.ones((max_sents), dtype=np.float64) / max_sents

            converged = False
            for n_iter in xrange(self.max_iters):
                last_r = r
                r = np.dot(M_hat, r)

                if (np.abs(r - last_r) < self.tol).any():
                    converged = True
                    break

            if not converged:
                print "warning:", 
                print "lexrank failed to converged after {} iters".format(
                    self.max_iters)
            input_df["f:lexrank"] = r
            return input_df, ndarray_data
        self._lexrank = lexrank

    def process(self, input_df, ndarray_data):
        return self._lexrank(input_df, ndarray_data)

    def requires(self):
        return []
    
    def ndarray_requires(self):
        return ["TfIdfCosSimMatrix",]

    def returns(self):
        return ["f:lexrank"]

    def ndarray_returns(self):
        return []

    def name(self):
        return "LexRankMixin"

class CentroidMixin(TfIdfMixin, BinaryBOWMixin):

    def build(self):
        pass

    def process(self, input_df, ndarray_data):
        B = ndarray_data["BinaryBOWMatrix"]
        X = ndarray_data["TfIdfMatrix"]
        c = X.sum(axis=0)
        assert c.shape[1] == X.shape[1]
        input_df["f:centroid"] = B.dot(c.T)
        return input_df, ndarray_data

    def requires(self):
        return []
    
    def ndarray_requires(self):
        return ["TfIdfMatrix", "BinaryBOWMatrix"]

    def returns(self):
        return ["f:centroid"]

    def ndarray_returns(self):
        return []

    def name(self):
        return "CentroidMixin"

class MMRMixin(TfIdfCosineSimilarityMixin):

    def build(self):
        if not hasattr(self, "lam"):
            self.lam = .7
        assert 0 < self.lam and self.lam < 1
        
        def rank(input_df, ndarray_data):
            K = ndarray_data["TfIdfCosSimMatrix"] 
            K = np.ma.masked_array(K, mask=np.diag(np.diag(K)))
            K_input = np.ma.masked_array(
                K, mask=False, fill_value=0, hardmask=False)
            K_summ = np.ma.masked_array(
                K, mask=True, fill_value=0, hardmask=False)

            w1 = self.lam
            w2 = (1 - w1)
            for rank in range(K.shape[0], 0, -1):
                if rank == K.shape[0]:
                    K_input_max = K_input.max(axis=1).filled(float("-inf"))
                    idx = np.argmax(K_input_max)
                else:
                    K_summ_max = K_summ.max(axis=1).filled(0) 
                    K_input_max = K_input.max(axis=1).filled(float("inf"))

                    S = w1 * K_summ_max - w2 * K_input_max
                    idx = np.argmax(S)

                K_summ.mask[:,idx] = False 
                K_summ.mask[idx, idx] = True
                K_input.mask[idx,:] = True

                input_df.ix[idx, "f:mmr"] = rank
                
            return input_df, ndarray_data
        self._mmr = rank

    def process(self, input_df, ndarray_data):
        return self._mmr(input_df, ndarray_data)

    def requires(self):
        return []
    
    def ndarray_requires(self):
        return ["TfIdfCosSimMatrix"]

    def returns(self):
        return ["f:mmr"]

    def ndarray_returns(self):
        return []

    def name(self):
        return "MMRMixin"
