import sumpy.io
import sumpy.system
from sumpy.simple import lede, centroid, mmr, textrank, lexrank



#import numpy as np
#from itertools import izip
#import nltk.data
#from nltk.tokenize import WordPunctTokenizer
#from nltk.stem.snowball import EnglishStemmer
#from nltk.corpus import stopwords
#import heapq
#from collections import defaultdict
#
#class DocumentSetReader(object):
#    def __init__(self, input=u"filename", preprocessor=None, sentence_processor=None, 
#            token_processor=None, token_processor_returns=None, stop_filter=None):
#
#        if input not in set([u"filename", u"file", u"content"]):
#            raise ValueError(
#                u"input argument must be 'filename', 'file', or 'content'") 
#        self.input = input
#
#        self.preprocessor = preprocessor
#
#        if sentence_processor is None:
#            senttok = nltk.data.load('tokenizers/punkt/english.pickle')
#            sentence_processor = lambda x: senttok.tokenize(x)  
#        self.sentence_processor = sentence_processor
#
#        if token_processor is None:
#            wordtok = WordPunctTokenizer()
#            stemmer = EnglishStemmer()
#            def default_token_processor(sentence):
#                tokens = [[stemmer.stem(word.lower())]
#                          for word in wordtok.tokenize(sentence)]
#                return tokens
#            token_processor = default_token_processor
#            token_processor_returns = ["token"]
#
#        self.token_processor = token_processor
#        self.token_processor_returns = token_processor_returns
#        
#        if stop_filter is None:
#            stop = stopwords.words('english')
#            stop_filter = lambda token: token in stop or len(token) <= 2
#        self.stop_filter = stop_filter
#        
#    def load_documents(self, documents, names=None):
#        max_docs = len(documents)
#        if names is None:
#            names = ["doc{}".format(n) for n in xrange(max_docs)]
#        assert len(names) == len(documents)    
#        
#        sentences = {}
#
#        token_type_index = self.token_processor_returns.index(u'token')
#        next_sentence_id = 0
#
#        for n_doc, (name, document) in enumerate(izip(names, documents)):
#            print n_doc
#            text = self._read(document)
#            for n_sent, sentence in enumerate(self.sentence_processor(text)):
#                tokens = {tok_type: list()
#                          for tok_type in self.token_processor_returns}
#                
#                for token_types in self.token_processor(sentence):
#                    if self.stop_filter(token_types[token_type_index]):
#                        continue
#                    for tok_type, token in izip(
#                        self.token_processor_returns, token_types):
#                        tokens[tok_type].append(token)
#                if len(tokens[u'token']) == 0:
#                    continue
#                
#                sentences[next_sentence_id] = {u"name": name,
#                                                    u"n_doc": n_doc,
#                                                    u"n_sent": n_sent,
#                                                    u"tokens": tokens,
#                                                    u"sentence": sentence}
#                next_sentence_id += 1
#        return sentences
#
#    def _read(self, document):
#
#        if self.input == u"filename":
#            with open(document, u"r") as f:
#                text = ''.join(f.readlines())
#        elif self.input == u"file":
#            text = ''.join(document.readlines())
#        elif self.input == u"content":
#            text = document
#
#        if isinstance(text, str):
#            text = text.decode(u"utf-8")
#        
#        if self.preprocessor is not None:
#            text = self.preprocessor(text)
#        
#        return text
#            
#class SentenceRanker(object):
#    pass
#
#class SumBasicRanker(SentenceRanker):
#
#    def rank(self, summary_input):
#        print "RANKING" 
#        ordered = []
#        unigram_probs = self._build_unigram_probs(summary_input)
#        
#        heap = [(1-prob, word) for word, prob in unigram_probs.items()]
#        heapq.heapify(heap)
#        
#        weights = []
#        token2sentids = defaultdict(list)
#        
#        covered = set()
#        n_sents = len(summary_input)
#
#        print "Debug"
#        for sent_id in sorted(summary_input.keys()):
#            weight = 0
#            length = 0
#            print sent_id
#            for token in summary_input[sent_id][u'tokens'][u'token']:
#                weight += unigram_probs[token]
#                token2sentids[token].append(sent_id)
#                length += 1
#                print u"{}/{}".format(token, weight),
#            print
#            weight /= float(length)
#            print weight
#            weights.append(weight)
#
#        while len(ordered) != n_sents:
#            # Get highest prob word (1)
#            prob, word = heapq.heappop(heap)
#            
#            # Get highest scored sentence containing highest prob word 
#            sent_ids = token2sentids[word]
#            sent_ids.sort(key=lambda x: weights[x])
#            
#            for sent_id in sent_ids:
#                print sent_id, weights[sent_id]
#                print summary_input[sent_id][u'sentence']
#            break
#
#            sent_id = sent_ids.pop()
#            while sent_id in covered:
#                if len(sent_ids) == 0:
#                    break
#                sent_id = sent_ids.pop()
#            
#            if len(sent_ids) == 0:
#                continue
#
#            ordered.append(sent_id)
#            covered.add(sent_id)
#        
#         #   for sent_id in sent_ids:
#         #       weights[sent_id] = (1 - prob)
#            heapq.heappush(heap, (1 - (1 - prob)**2, word))
#            print word, weights
#            print summary_input[sent_id][u'sentence']
#            #for sent_id in sent_ids:
#            #    print sent_id, weights[sent_id]
#            #for prob, word in heapq.heappop(heap)    
#    #def 
#
#    def _build_unigram_probs(self, summary_input):
#        probs = {}
#        total = 0
#        for sentence in summary_input.values():
#            for token in sentence[u'tokens'][u'token']:
#                probs[token] = probs.get(token, 0) + 1
#                total += 1
#
#        assert total > 1
#        total = float(total)
#        for key in probs.keys():
#            probs[key] /= total
#        return probs
#
#class PageRank(object):
#
#    def __init__(self, max_iters=100, tol=1E-4, d=.85):
#        self.max_iters = max_iters
#        self.tol = tol
#        self.d = d
#
#    def rank(self, K):
#        n_nodes = K.shape[0]
#        r = np.ones((n_nodes, 1), dtype=np.float64) / n_nodes
#        #r /= np.sum(r)
#        last_r = np.ones((n_nodes, 1))
#        K_hat = (self.d * K) + \
#            (float(1 - self.d) / n_nodes) * np.ones((n_nodes, n_nodes))
#        
#        converged = False
#        for n_iter in xrange(self.max_iters):
#            last_r = r
#            r = np.dot(K_hat, r)
#            r /= np.sum(r)
#
#            if (np.abs(r - last_r) < self.tol).any():
#                converged = True
#                break
#
#        if not converged:
#            print "Warning: PageRank not converged after %d iters" % self.max_iters
#
#        return r
#
#
#
#class LexRank(object): 
#    pass
#
#class TextRank(object):
#
#    def summarize(self, text_units):
#        pass
#
#    def sentence_tokenizer(self):
#        pass
#    def word_tokenizer(self):
#        pass
