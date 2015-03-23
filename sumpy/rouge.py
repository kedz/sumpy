from nltk.util import ngrams
from sumpy.preprocessor import (SentenceTokenizerMixin, 
    ROUGEWordTokenizerMixin, SMARTStopWordsMixin)
import pandas as pd

class ROUGE(SentenceTokenizerMixin, ROUGEWordTokenizerMixin, 
            SMARTStopWordsMixin):
    def __init__(self, sentence_tokenizer=None, word_tokenizer=None, 
                 max_ngrams=2, remove_stopwords=False, stopwords=None,
                 show_per_model_results=False):
        
        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer
        self._max_ngrams = max_ngrams
        self.remove_stopwords = remove_stopwords
        self._stopwords = stopwords
        self._show_per_model_results = show_per_model_results

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

        # Collect results as a pandas DataFrame and compute the mean 
        # performance.
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
        df2 = df.groupby(level=0).mean()
        if self._show_per_model_results is True:
            df2['model'] = 'AVG'
            df2 = df2.reset_index().set_index(['system','model']).append(df)
            df2 = df2.sort()
        
        return df2

    def extract_ngrams(self, text, sent_tokenizer, word_tokenizer, max_ngrams,
            is_stopword):
        ngram_sets = {}
        sents = sent_tokenizer(text)
        
        tokens = []
        for sent in sents:
            tokens.extend([word.lower() for word in word_tokenizer(sent)])

        # Remove stopwords.
        tokens = [word for word in tokens if is_stopword(word) is False]
        
        for i in xrange(1, max_ngrams + 1):
            ngram_sets[i] = {}
            total = 0
            for ngram in ngrams(tokens, i):
                ngram_sets[i][ngram] = ngram_sets[i].get(ngram, 0) + 1
                total += 1
            ngram_sets[i][u"__TOTAL__"] = total
        return ngram_sets
                      
    def compute_prf(self, sys_ngram_sets, model_ngram_sets, max_ngrams):
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

            if intersect == 0: 
                print "Warning: 0 {}-gram overlap".format(i)
                f1 = 0
            else:
                f1 = 2 * prec * recall / (prec + recall)
            scores.append(recall)
            scores.append(prec)
            scores.append(f1)
            
        return scores
