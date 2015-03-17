import nltk.data
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import gzip
import pkg_resources
import os

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


class ROUGEWordTokenizerMixin(object):
    def build_word_tokenizer(self):
        """This mixin provides the same reg-ex based word tokenizer that is
        used in the official ROUGE perl script (Lin, 2004). See the readText 
        subroutine (line 1816) of ROUGE-1.5.5.pl for reference."""
        if self._word_tokenizer is not None:
            tokenize = self._word_tokenizer
        else:
            def rouge_tokenize(sentence):                  
                s = re.sub(r"-", r" -", sentence, flags=re.UNICODE)
                s = re.sub(r"[^A-Za-z0-9\-]", r" ", s, flags=re.UNICODE) 
                s = s.strip()
                s = re.sub(r"\s+", r" ", s, flags=re.UNICODE)
                return s.split(u" ") 
            tokenize = rouge_tokenize
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

class SMARTStopWordsMixin(object):
    def build_stopwords(self):
        if self.remove_stopwords is True:
            if self._stopwords is None:             
                path = pkg_resources.resource_filename(
                    "sumpy", 
                    os.path.join("data", "smart_common_words.txt.gz"))
                with gzip.open(path, u"r") as f:
                    self._stopwords = set(
                        [word.strip().decode(u"utf-8").lower()
                         for word in f.readlines()])
            return lambda word: word in self._stopwords
        else:
            return lambda word: False
