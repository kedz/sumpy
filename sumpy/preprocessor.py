import nltk.data
from nltk.tokenize import WordPunctTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import gzip
import pkg_resources
import os

class NamedEntityRecogMixin(object):
    def build_named_entity_recog(self):
        """Return a funcion that returns a tree with 
        named entities recognized in subtrees"""
        if self._named_entity_recog is not None:
            ne = self._named_entity_recog
        else: 
            ne = nltk.ne_chunk
        return ne

class WordLemmatizerMixin(object):
    def build_word_lemmatizer(self):
        """Return a function that lemmatizes a word
        given a word and its pos."""
        if self._word_lemmatizer is not None:
            lem = self._word_lemmatizer
        else:
            wordnet_lemmatizer = WordNetLemmatizer()
            lem = wordnet_lemmatizer.lemmatize
        return lem

class PosTaggerMixin(object):
    def build_pos_tag(self):
        """Return a function that returns the part of
        speech of a given tokenized word sentence"""
        if self._pos_tag is not None:
            pos = self._pos_tag
        else: 
            pos = nltk.pos_tag
#data.load('taggers/maxent_treebank_pos_tagger/english.pickle').pos_tag
        return pos

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

class LengthLimiterMixin(object):
    def build_length_limiter(self):
        """
            Return a function that shortens a list of tokens to a 
            desired length.
        """
        if self._limit is None and self._limit_type is not None:
            raise Exception("Both limit and limit_type must be set.")
        if self._limit is not None and self._limit_type is None:
            raise Exception("Both limit and limit_type must be set.")
        if self._limit_type not in [None, u"word"]:
            raise Exception(
                "limit_type: {} not implemented.".format(self._limit_type))
        
        if self._limit_type is None:
            # Do not shorten, just return tokens unchanged.
            return lambda x: x
        if self._limit_type == u"word":
            # Shorten list to be `_limit` tokens long.
            def word_limiter(sequence):
                if len(sequence) < self._limit:
                    print "Warning: document is shorter than the max length" \
                          + " limit. This can effect evaluation negatively."
                return sequence[:self._limit]
            return word_limiter
