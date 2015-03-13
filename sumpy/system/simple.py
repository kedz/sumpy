from sumpy.preprocessor import (SentenceTokenizerMixin, WordTokenizerMixin,
    CorpusTfidfMixin)
from sumpy.rankers import (LedeRankerMixin, TextRankMixin, LexRankMixin, 
    CentroidScoreMixin)
from sumpy.document import Summary
import pandas as pd

class LedeSummarizer(SentenceTokenizerMixin, LedeRankerMixin):
    def __init__(self, sentence_tokenizer=None):
        self._sentence_tokenizer = sentence_tokenizer
    
    def summarize(self, docs):
        analyze = self.build_sent_tokenizer()
        docs = [analyze(doc) for doc in docs]
        
        sents = []
        for doc_no, doc in enumerate(docs, 1):
            for sent_no, sent in enumerate(doc, 1):
                sents.append({"doc": doc_no, "doc position": sent_no, 
                              "text": sent})
        input_df = pd.DataFrame(sents,
                                columns=["doc", "doc position", "text", 
                                         "rank:lede"])

        self.rank_by_lede(input_df)
        summary_df = input_df.loc[input_df["rank:lede"] == 1]
        return Summary(summary_df)
       
class TextRankSummarizer(SentenceTokenizerMixin, WordTokenizerMixin, 
                         TextRankMixin):
    def __init__(self, sentence_tokenizer=None, word_tokenizer=None):
        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer 

    def summarize(self, docs):
        sent_tokenize = self.build_sent_tokenizer()
        word_tokenize = self.build_word_tokenizer()
        docs = [sent_tokenize(doc) for doc in docs]
        
        sents = []
        for doc_no, doc in enumerate(docs, 1):
            for sent_no, sent in enumerate(doc, 1):
                words = word_tokenize(sent)
                sents.append({"doc": doc_no, "doc position": sent_no, 
                    "text": sent, "words": words})
        input_df = pd.DataFrame(sents,
                                columns=["doc", "doc position", "text", 
                                         "words", "rank:textrank"])

        self.textrank(input_df)
        input_df.sort(["rank:textrank"], inplace=True, ascending=False)
        return Summary(input_df)
       
class LexRankSummarizer(SentenceTokenizerMixin, WordTokenizerMixin, 
                        CorpusTfidfMixin, LexRankMixin):
    def __init__(self, sentence_tokenizer=None, word_tokenizer=None):
        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer 

    def summarize(self, docs):
        sent_tokenize = self.build_sent_tokenizer()
        word_tokenize = self.build_word_tokenizer()
        tfidfer = self.build_tfidf_vectorizer()
        docs = [sent_tokenize(doc) for doc in docs]
        
        sents = []
        for doc_no, doc in enumerate(docs, 1):
            for sent_no, sent in enumerate(doc, 1):
                words = word_tokenize(sent)
                sents.append({"doc": doc_no, "doc position": sent_no, 
                    "text": sent, "words": words})
        input_df = pd.DataFrame(sents,
                                columns=["doc", "doc position", "text", 
                                         "words", "rank:lexrank"])

        tfidf_mat = tfidfer(input_df[u"words"].tolist())
        self.lexrank(input_df, tfidf_mat)
        input_df.sort(["rank:lexrank"], inplace=True, ascending=False)
        return Summary(input_df)


class CentroidSummarizer(SentenceTokenizerMixin, WordTokenizerMixin,
                         CorpusTfidfMixin, CentroidScoreMixin):
    def __init__(self, sentence_tokenizer=None, word_tokenizer=None):
        self._sentence_tokenizer = sentence_tokenizer
        self._word_tokenizer = word_tokenizer
    
    def summarize(self, docs):
        sent_tokenize = self.build_sent_tokenizer()
        word_tokenize = self.build_word_tokenizer()
        tfidfer = self.build_tfidf_vectorizer()
        docs = [sent_tokenize(doc) for doc in docs]
        
        sents = []
        for doc_no, doc in enumerate(docs, 1):
            for sent_no, sent in enumerate(doc, 1):
                words = word_tokenize(sent)
                sents.append({"doc": doc_no, "doc position": sent_no, 
                    "text": sent, "words": words})
        input_df = pd.DataFrame(sents,
                                columns=["doc", "doc position", "text", 
                                         "words", "rank:centroid_score"])

        tfidf_mat = tfidfer(input_df[u"words"].tolist())
        self.centroid_score(input_df, tfidf_mat)

        input_df.sort([u"rank:centroid_score"], inplace=True, ascending=False)
        return Summary(input_df)
