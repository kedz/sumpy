from nltk.util import ngrams
from sumpy.preprocessor import (SentenceTokenizerMixin, 
    ROUGEWordTokenizerMixin, SMARTStopWordsMixin)
import pandas as pd

class ROUGE(SentenceTokenizerMixin, ROUGEWordTokenizerMixin, 
            SMARTStopWordsMixin):
    def __init__(self, sentence_tokenizer=None, word_tokenizer=None, 
                 max_ngrams=2, remove_stopwords=False, stopwords=None):
        
        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer
        self._max_ngrams = max_ngrams
        self.remove_stopwords = remove_stopwords
        self._stopwords = stopwords

    def evaluate(self, systems, models):
        models = list(models) # make model order consistent
        sent_tokenizer = self.build_sent_tokenizer()
        word_tokenizer = self.build_word_tokenizer()
        is_stopword = self.build_stopwords()
        results = []
        result_index = []
        for name, system in systems:
            sys_ngram_sets = self.extract_ngrams(
                system, sent_tokenizer, word_tokenizer, self._max_ngrams,
                    is_stopword)

            for model_no, model in enumerate(models, 1):
                model_ngram_sets = self.extract_ngrams(
                    model, sent_tokenizer, word_tokenizer, self._max_ngrams,
                    is_stopword)
                scores = self.compute_prf(
                    sys_ngram_sets, model_ngram_sets, self._max_ngrams)
                result_index.append((name, model_no))
                results.append(scores)

        col_index = []
        dataframe_cols = []
        for i in xrange(1, self._max_ngrams + 1):
            rouge_n = u"ROUGE-{}".format(i)
            col_index.append((rouge_n, "Recall"))
            col_index.append((rouge_n, "Prec."))
            col_index.append((rouge_n, "F1"))
        
        row_index = pd.MultiIndex.from_tuples(
            result_index, names=['system', 'model'])
        col_index = pd.MultiIndex.from_tuples(col_index)
        df = pd.DataFrame(results, columns=col_index, index=row_index)
        
        return df

    def extract_ngrams(self, text, sent_tokenizer, word_tokenizer, max_ngrams,
            is_stopword):
        ngram_sets = {}
        sents = sent_tokenizer(text)
        
        #sents = [[word.lower() for word in word_tokenizer(sent)]
        #            for sent in sents]
        tokens = []
        for sent in sents:
            tokens.extend([word.lower() for word in word_tokenizer(sent)])


        # Remove stopwords.
        tokens = [word for word in tokens if is_stopword(word) is False]
        
        for i in xrange(1, max_ngrams + 1):
            ngram_sets[i] = {}
            total = 0
            #for sent in sents:
            #for ngram in ngrams(sent, i):
            for ngram in ngrams(tokens, i):
                ngram_sets[i][ngram] = ngram_sets[i].get(ngram, 0) + 1
                total += 1
            ngram_sets[i][u"__TOTAL__"] = total
        return ngram_sets
                      
    def compute_prf(self, sys_ngram_sets, model_ngram_sets, max_ngrams):
       # scores = {}
        scores = []
        for i in xrange(1, max_ngrams + 1):
            intersect = 0
            for ngram, model_ngram_count in model_ngram_sets[i].items():
                if ngram == "__TOTAL__":
                    continue
                sys_ngram_count = sys_ngram_sets[i].get(ngram, 0)
                intersect += min(model_ngram_count, sys_ngram_count)
            recall = float(intersect) / model_ngram_sets[i][u"__TOTAL__"]
            prec = float(intersect) / sys_ngram_sets[i][u"__TOTAL__"]
            #intersect = sys_ngram_sets[i].intersection(model_ngram_sets[i])
            #n_intersect = len(intersect)
            #recall = float(n_intersect) / len(model_ngram_sets[i])
            #prec = float(n_intersect) / len(sys_ngram_sets[i])
            f1 = 2 * prec * recall / (prec + recall)
            scores.append(recall)
            scores.append(prec)
            scores.append(f1)
            #scores[u"ROUGE-{} Recall".format(i)] = recall
            #scores[u"ROUGE-{} Prec.".format(i)] = prec
            #scores[u"ROUGE-{} F1".format(i)] = f1
            
        return scores
