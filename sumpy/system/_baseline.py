from sumpy.system._base import _SystemBase
from sumpy.annotators import (WordTokenizerMixin, LedeMixin, MMRMixin, 
        CentroidMixin)
from sumpy.document import Summary

class LedeSummarizer(WordTokenizerMixin, LedeMixin, _SystemBase):

    def __init__(self, sentence_tokenizer=None, word_tokenizer=None,
            verbose=False):
        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer
        super(LedeSummarizer, self).__init__(verbose=verbose)

    def build_summary(self, input_df, ndarray_data):
        output_df = input_df[input_df[u"f:lede"] == 1].sort(
            ["doc id"], ascending=True)
        return Summary(output_df)

class CentroidSummarizer(CentroidMixin, _SystemBase):

    def __init__(self, sentence_tokenizer=None, word_tokenizer=None,
            verbose=False):
        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer
        super(CentroidSummarizer, self).__init__(verbose=verbose)

    def build_summary(self, input_df, ndarray_data):
        output_df = input_df.sort(["f:centroid"], ascending=False)
        return Summary(output_df)

class MMRSummarizer(MMRMixin, _SystemBase):
    def __init__(self, sentence_tokenizer=None, word_tokenizer=None,
            lam=.4, verbose=False):
        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer
        self.lam = lam
        super(MMRSummarizer, self).__init__(verbose=verbose)
        
    def build_summary(self, input_df, ndarray_data):
        output_df = input_df.sort(["f:mmr"], ascending=False)
        return Summary(output_df)
