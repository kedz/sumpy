from sklearn.metrics import pairwise_distances
from duc_testbed import load_docsets
from sumpy.rankers import ConceptMixin
from sumpy.preprocessor import (SentenceTokenizerMixin, WordTokenizerMixin, CorpusTfidfMixin) 
import argparse
import pandas as pd
import os
import sumpy
import sumpy.eval
import numpy as np
from sumpy.eval import ROUGE

class ROUGELoss(ROUGE, SentenceTokenizerMixin, WordTokenizerMixin):
    def __init__(self, all_docs, model, ngrams=1, sentence_tokenizer=None, \
        word_tokenizer=None, remove_stopwords=False, stopwords=None,
        limit=None, limit_type=None):
        self._limit = limit
        self._limit_type = limit_type
        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer
        self._max_ngrams = ngrams
        self.remove_stopwords = remove_stopwords
        self._stopwords = stopwords
        self._show_per_model_results = False
        sent_tokenizer = self.build_sent_tokenizer()
        word_tokenizer = self.build_word_tokenizer()
        length_limiter = self.build_length_limiter()
        is_stopword = self.build_stopwords()
        #get ngrams in model
        self._model_ngrams = self.extract_ngrams(model, sent_tokenizer, \
            word_tokenizer, self._max_ngrams, is_stopword, length_limiter)
        #get ngrams in all documents
        self._all_ngrams = []
        self._denominator = 0
        for doc in all_docs:
            doc_ngrams = self.extract_ngrams(doc, sent_tokenizer, \
                word_tokenizer, self._max_ngrams, is_stopword, length_limiter)
            #remove ngrams in model from doc's ngrams
            for key in self._model_ngrams[self._max_ngrams]:
                if key in doc_ngrams[self._max_ngrams]:
                    del doc_ngrams[self._max_ngrams][key]
            #increment value of ROUGELoss's denominator value
            for value in doc_ngrams[self._max_ngrams].values():
                self._denominator += value
    
    def get_rouge_loss(self, text):
        sent_tokenizer = self.build_sent_tokenizer()
        word_tokenizer = self.build_word_tokenizer()
        length_limiter = self.build_length_limiter()
        is_stopword = self.build_stopwords()
        text_ngrams = self.extract_ngrams(text, sent_tokenizer, \
            word_tokenizer, self._max_ngrams, is_stopword, length_limiter)
        for key in self._model_ngrams[self._max_ngrams]:
            if key in text_ngrams[self._max_ngrams]:
                del text_ngrams[self._max_ngrams][key]
        numerator = 0
        for value in text_ngrams[self._max_ngrams].values():
            numerator += value
        result = numerator * 1.0 / self._denominator
        #if not result == 0:
            #print 'loss: ', result
        return result
        
class fLAI(ConceptMixin, SentenceTokenizerMixin, CorpusTfidfMixin):
    def __init__(self, all_docs, model, weights, functions, alphas,
        docset_id, sentence_tokenizer=None):
        self._loss = ROUGELoss(all_docs, model)
        self.model = model 
        self.weights = weights
        self.functions = functions
        self.alphas = alphas
        self.docset_id = docset_id

    def get_cos_dis(self, input_df):
        tfidfer = self.build_tfidf_vectorizer()
        self.tfidfs = tfidfer(input_df['words'].tolist())
        self.cos_dis = 1-pairwise_distances(self.tfidfs, metric="cosine")

    def set_concepts(self, input_df):
        self.conceptrank(input_df, self.docset_id)
        self.input_df = input_df
    
    def rank(self, indices, ignore, l=None):
        text = ''
        for index in indices:
            text += ' ' + self.input_df['text'][index]
        f_t = self.get_trunc_vector(indices)
        dot_prod = 0
        for i in range(0, len(f_t)):
            dot_prod += self.weights[i] * f_t[i] 
        return dot_prod + self._loss.get_rouge_loss(text)
    
    def get_trunc_vector(self, indices, model=False, p=False):
        vector = np.zeros(len(self.weights))
        index = 0
        for function in self.functions:
            normal_return, gold_return = function(self, indices, model)
            if p:
                print 'indices: ', indices, ' model: ', model
                print 'normal_return: ', normal_return, ' total_return: ', gold_return
            for alpha in self.alphas:
                frac_return = alpha * gold_return
                vector[index] = normal_return
                if frac_return < normal_return:
                    vector[index] = frac_return
                index += 1
        return vector

    def get_concept_counts(self, indices, model=False):
        binary_concepts = self.binary_concepts
        if len(binary_concepts) == 0:
            return 0, 0
        total_concepts = len(self.concept_sizes.keys())
        model_count = self._num_model_concepts 
        if model:
            return model_count, total_concepts
        current_concepts = {}
        concept_count = 0
        for sent_index in indices:
            for concept in binary_concepts[sent_index].keys():
                if not concept in current_concepts.keys():
                    current_concepts[concept] = 1
                    concept_count += 1
        return concept_count, total_concepts

def learn_submod_shells(docsets, functions, alphas, learning_rate = 0.01, my_lambda=0.01): #Rona needs to finish params
    #initialize weights to 0
    ws = []
    w = np.zeros(len(functions) * len(alphas))
    ws.extend(w)
    #Go throuogh each docset
    for t, docset_id in enumerate(docsets.keys()):
        #Get learning rate
        eta = learning_rate
        if isinstance(learning_rate, list):
            eta = learning_rate[t]
        #docs and model for this set 
        docs = docsets[docset_id][u'docs']
        model = docsets[docset_id][u'model']
        #Run greedy reranker with fLAI
        s = sumpy.system.RerankerSummarizer()
        class_ranker = fLAI(docs, model, w, functions, alphas, docset_id)
        word_count = int((len(model.split(' ')) + 100) / 100) * 100
        subtrahend = word_count / 2
        word_count -= subtrahend
        summary = s.summarize(docs, model, class_ranker, word_count)
        results = class_ranker._loss.evaluate([('submodular', unicode(summary))], [model])
        print results
        max_indices = s.indices
        print 'finished a round of greedy reranker'
        #Get g_t
        f_t_y = class_ranker.get_trunc_vector(max_indices, False, True)
        print 'f_t_y: ', f_t_y  
        f_t_gold = class_ranker.get_trunc_vector(None, True, True)
        print 'f_t_gold: ', f_t_gold
        g_t = my_lambda * w + f_t_y - f_t_gold
        print 'g_t: ', g_t
        #Check if each w_i is > 0
        w = np.maximum(0, w - eta * g_t) 
       # for i, w_i in enumerate(w):
       #     new_w_i = w_i - eta * g_t[i]
       #     if new_w_i < 0:
       #         w[i] = 0
       #     else:
       #         w[i] = new_w_i
        #Add w to my list of ws        
        print 'weights: ', w
        ws.extend(w)
    #Return average of ws      
    return (1.0 / len(ws)) * sum(ws)

def create_doc_model(docsets):
    new_docsets = {}
    length = len(docsets.keys())
    for docset_id in docsets.keys():
        docs = docsets[docset_id][u'docs']
        for i, model in enumerate(docsets[docset_id][u'models']):
            new_docset_id = str(i * length) + docset_id
            new_docsets[new_docset_id] = {}
            new_docsets[new_docset_id][u'docs'] = docs
            new_docsets[new_docset_id][u'model'] = model 
    return new_docsets

def main(duc_dir):
    print u"Loading DUC document sets from:", duc_dir 
    docsets = load_docsets(duc_dir)
    docsets = create_doc_model(docsets)
    alphas = np.array(range(1,11)) / 100.0
    functions = [fLAI.get_concept_counts] 
    print u'Done loading'
    return learn_submod_shells(docsets, functions, alphas)

if __name__ == u"__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(u"-d", u"--duc-dir", required=True, type=unicode,
                        help=u"path to DUC document set directory")
    args = parser.parse_args()
    duc_dir = args.duc_dir
    main(duc_dir)

