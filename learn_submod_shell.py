from duc_testbed import load_docsets
from sumpy.rankers import ConceptMixin
from sumpy.preprocessor import (SentenceTokenizerMixin, WordTokenizerMixin) 
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
        
class fLAI(ConceptMixin, SentenceTokenizerMixin):
    def __init__(self, all_docs, model, weights, functions, alphas,
        sentence_tokenizer=None):
        self._loss = ROUGELoss(all_docs, model)
        self.model = model 
        self.weights = weights
        self.functions = functions
        self.alphas = alphas
    
    def set_concepts(self, input_df):
        self.conceptrank(input_df)
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
    
    def get_trunc_vector(self, indices, model=False):
        vector = np.zeros(len(self.weights))
        index = 0
        for function in self.functions:
            normal_return, gold_return = function(self, indices, model)
            for alpha in self.alphas:
                frac_return = alpha * gold_return
                vector[index] = normal_return
                if frac_return < normal_return:
                    vector[index] = frac_return
                index += 1
        return vector

    def get_concept_counts(self, indices, model=False):
        binary_concepts = self.get_binary_concepts()
        if len(binary_concepts) == 0:
            return 0, 0
        total_concepts = len(binary_concepts[0])
        model_count = self._num_model_concepts 
        if model:
            return model_count, model_count
        current_concepts = np.zeros(total_concepts)
        concept_count = 0
        for sent_index in indices:
            for concept_index in range(0, total_concepts):
                if binary_concepts[sent_index][concept_index] \
                    and not current_concepts[concept_index]:
                    current_concepts[concept_index] = 1
                    concept_count += 1
        return concept_count, model_count

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
        class_ranker = fLAI(docs, model, w, functions, alphas)
        s.summarize(docs, model, class_ranker, len(model))
        max_indices = s.indices
        print 'finished a round of greedy reranker'
        #Get g_t
        f_t_y = class_ranker.get_trunc_vector(max_indices)
        print 'f_t_y: ', f_t_y  
        f_t_gold = class_ranker.get_trunc_vector(None, True)
        g_t = my_lambda * w + f_t_y - f_t_gold
        #Check if each w_i is > 0
        for i, w_i in enumerate(w):
            new_w_i = w_i - eta * g_t[i]
            if new_w_i < 0:
                w[i] = 0
            else:
                w[i] = new_w_i
        #Add w to my list of ws        
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
    alphas = np.array(range(1,11)) / 10.0
    functions = [fLAI.get_concept_counts] 
    return learn_submod_shells(docsets, functions, alphas)

if __name__ == u"__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(u"-d", u"--duc-dir", required=True, type=unicode,
                        help=u"path to DUC document set directory")
    args = parser.parse_args()
    duc_dir = args.duc_dir
    main(duc_dir)

