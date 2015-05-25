from sumpy.system._base import _SystemBase
from sumpy.annotators import TextRankMixin, LexRankMixin
from sumpy.document import Summary

class TextRankSummarizer(TextRankMixin, _SystemBase):

    def __init__(self, sentence_tokenizer=None, word_tokenizer=None, 
            directed=u"undirected", d=.85, tol=.0001, max_iters=20, 
            verbose=False):
        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer
        self.directed = directed
        self.d = d
        self.tol = tol
        self.max_iters = max_iters
        super(TextRankSummarizer, self).__init__(verbose=verbose)

    def build_summary(self, input_df, ndarray_data):
        output_df = input_df.sort(["f:textrank"], ascending=False)
        return Summary(output_df)

class LexRankSummarizer(LexRankMixin, _SystemBase):

    def __init__(self, sentence_tokenizer=None, word_tokenizer=None, 
            d=.85, tol=.0001, max_iters=20, 
            verbose=False):
        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer
        self.d = d
        self.tol = tol
        self.max_iters = max_iters
        super(LexRankSummarizer, self).__init__(verbose=verbose)

    def build_summary(self, input_df, ndarray_data):
        output_df = input_df.sort(["f:lexrank"], ascending=False)
        return Summary(output_df)

