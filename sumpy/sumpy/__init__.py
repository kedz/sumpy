import numpy as np

class PageRank(object):

    def __init__(self, max_iters=100, tol=1E-4, d=.85):
        self.max_iters = max_iters
        self.tol = tol
        self.d = d

    def rank(self, K):
        n_nodes = K.shape[0]
        r = np.random.uniform(size=(n_nodes, 1))
        r /= np.sum(r)
        last_r = np.ones((n_nodes, 1))
        K_hat = (self.d * K) + \
            (float(1 - self.d) / n_nodes) * np.ones((n_nodes, n_nodes))
        
        converged = False
        for n_iter in xrange(self.max_iters):
            last_r = r
            r = np.dot(K_hat, r)
            r /= np.sum(r)

            if (np.abs(r - last_r) < self.tol).any():
                converged = True
                break

        if not converged:
            print "Warning: PageRank not converged after %d iters" % self.max_iters

        return r

class TextRank(object):

    def summarize(self, text_units):
        pass

    def sentence_tokenizer(self):
        pass
    def word_tokenizer(self):
        pass
