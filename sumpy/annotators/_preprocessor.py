from sumpy.annotators._annotator_base import _AnnotatorBase
import pkg_resources
import gzip
import os
import pandas as pd
import nltk
from nltk.tokenize import WordPunctTokenizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re


class SentenceTokenizerMixin(_AnnotatorBase):
    """
    Analyze method takes a string (an article text usually) and splits it
    into substrings corresponding to the sentences in the origial article.
    """

    def requires(self):
        return ["doc text"]
    
    def ndarray_requires(self):
        return []

    def returns(self):
        return ["sent id", "sent text"]

    def ndarray_returns(self):
        return []

    def name(self):
        return "SentenceTokenizerMixin"

    def build(self):

        if not hasattr(self, "_sentence_tokenizer"):
            self._sentence_tokenizer = None

        if self._sentence_tokenizer is None:
            dl = nltk.downloader.Downloader()
            if dl.is_installed("punkt") is False:
                print "Installing NLTK Punkt Sentence Tokenizer"
                dl.download("punkt")

            self._sentence_tokenizer = nltk.data.load(
                'tokenizers/punkt/english.pickle').tokenize

    def process(self, input_df, ndarray_data):
        def split_text(group):
            row = group.irow(0)
            sents = self._sentence_tokenizer(row["doc text"])
            return pd.DataFrame([{"doc id": row["doc id"], 
                                  "sent id": i, "sent text": sent}
                                  for i, sent in enumerate(sents, 1)])

        processed_df = input_df.groupby(
            "doc id", group_keys=False).apply(split_text) 
       
        cols = input_df.columns.difference(processed_df.columns).tolist()
        cols += ["doc id"]
        output_df = input_df[cols].merge(
            processed_df, on="doc id", how="inner")
        return output_df, ndarray_data

class WordTokenizerMixin(SentenceTokenizerMixin):
    """Analyze method takes a string (corresponding to a sentence) and splits
it into substrings corresponding to the words in original aritcle."""
    
    def build(self):

        if not hasattr(self, "_word_tokenizer"):
            self._word_tokenizer = None

        if self._word_tokenizer is None:
            self._word_tokenizer = WordPunctTokenizer().tokenize

    def process(self, input_df, ndarray_data):
        input_df["words"] = input_df["sent text"].apply(
            self._word_tokenizer)
        return input_df, ndarray_data

    def requires(self):
        return ["sent id", "sent text"]
   
    def ndarray_requires(self):
        return []

    def returns(self):
        return ["words"]

    def ndarray_returns(self):
        return []

    def name(self):
        return "WordTokenizerMixin"

class RawBOWMixin(WordTokenizerMixin):

    def build(self):

        if not hasattr(self, "_count_vectorizer"):
            self._count_vectorizer = None

        if self._count_vectorizer is None:
            self._count_vectorizer = CountVectorizer(
                    input=u"content", preprocessor=lambda x: x,
                    tokenizer=lambda x: x)

    def process(self, input_df, ndarray_data):
        ndarray_data["RawBOWMatrix"] = self._count_vectorizer.fit_transform(
            input_df["words"].tolist())
        return input_df, ndarray_data

    def requires(self):
        return ["words"]
    
    def returns(self):
        return []

    def ndarray_requires(self):
        return []

    def ndarray_returns(self):
        return ["RawBOWMatrix"]

    def name(self):
        return "RawBOWMixin"
       
class BinaryBOWMixin(RawBOWMixin):

    def build(self):
        pass

    def process(self, input_df, ndarray_data):
        X = ndarray_data["RawBOWMatrix"].copy()
        X[X > 0] = 1
        ndarray_data["BinaryBOWMatrix"] = X
        return input_df, ndarray_data

    def requires(self):
        return []
    
    def returns(self):
        return []

    def ndarray_requires(self):
        return ["RawBOWMatrix",]

    def ndarray_returns(self):
        return ["BinaryBOWMatrix"]

    def name(self):
        return "BinaryBOWMixin"

class TfIdfMixin(RawBOWMixin):
    def build(self):
        if not hasattr(self, "_tfidf_transformer"):
            self._tfidf_transformer = None

        if self._tfidf_transformer is None:
            self._tfidf_transformer = TfidfTransformer()
                    #input=u"content", preprocessor=lambda x: x,
                    #tokenizer=lambda x: x)

    def process(self, input_df, ndarray_data):
        X = self._tfidf_transformer.fit_transform(
            ndarray_data["RawBOWMatrix"])
        ndarray_data["TfIdfMatrix"] = X
        return input_df, ndarray_data

    def requires(self):
        return []

    def returns(self):
        return []

    def ndarray_requires(self):
        return ["RawBOWMatrix",]

    def ndarray_returns(self):
        return ["TfIdfMatrix"]

    def name(self):
        return "TfIdfMixin"

class TfIdfCosineSimilarityMixin(TfIdfMixin):

    def build(self):
        pass

    def process(self, input_df, ndarray_data):
        K = cosine_similarity(ndarray_data["TfIdfMatrix"])
        ndarray_data["TfIdfCosSimMatrix"] = K
        return input_df, ndarray_data

    def requires(self):
        return []

    def returns(self):
        return []

    def ndarray_requires(self):
        return ["TfIdfMatrix"]

    def ndarray_returns(self):
        return ["TfIdfCosSimMatrix"] 

    def name(self):
        return "TfIdfCosineSimilarityMixin"


