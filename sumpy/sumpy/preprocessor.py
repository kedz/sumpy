import nltk.data
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer

class SentenceTokenizerMixin(object):

    def build_sent_tokenizer(self):
        """Return a function that splits a string into a sequence of 
        sentences."""
        if self._sentence_tokenizer is not None:
            tok = self._sentence_tokenizer
        else:
            tok = nltk.data.load('tokenizers/punkt/english.pickle').tokenize
        return tok


class WordTokenizerMixin(object):
    def build_word_tokenizer(self):
        """Return a function that splits a string into a sequence of words."""
        if self._word_tokenizer is not None:
            tokenize = self._word_tokenizer
        else:
            tokenize = WordPunctTokenizer().tokenize
        return tokenize

class CorpusTfidfMixin(object):
    def build_tfidf_vectorizer(self):
        self._tfidf_vectorizer = TfidfVectorizer(analyzer=lambda x: x)
        return self._tfidf_vectorizer.fit_transform

class TextAnalyzerMixin(object):

    def build_analyzer(self):
        sent_tokenize = self._build_sent_tokenizer()
        word_tokenize = self._build_word_tokenizer()
        stem = self._build_stemmer()
        def analyzer(text):
            sents = sent_tokenize(text)
            tokenized_sents = [[stem(word) for word in word_tokenize(sent)]
                               for sent in sents]
            return tokenized_sents, sents
        return analyzer

    def _build_sent_tokenizer(self):
        """Return a function that splits a string into a sequence of 
        sentences."""
        if self._sentence_tokenizer is not None:
            return self._sentence_tokenizer
        else:
            return nltk.data.load('tokenizers/punkt/english.pickle').tokenize

    def _build_word_tokenizer(self):
        """Return a function that splits a string into a sequence of words."""
        if self._word_tokenizer is not None:
            tokenize = self._word_tokenizer
        else:
            tokenize = WordPunctTokenizer().tokenize

        return tokenize

    def _build_stemmer(self):
        if self._stemmer is not None:
            return self._stemmer
        else: return lambda w: w


