import re
import textwrap
from sumpy.preprocessor import WordTokenizerMixin

class Summary(WordTokenizerMixin):
    def __init__(self, df):
        self._df = df
        self._word_tokenizer = None
        self._tokenize = self.build_word_tokenizer()

    def _pretty_sent(self, sentence):
        no_rn = re.sub(r"\n|\r", r" ", sentence, flags=re.UNICODE)
        return re.sub(r"  \s+", r" ", no_rn, flags=re.UNICODE)

    def _make_pretty(self, length=200):
        size = 0
        sents = []
        for sent in self._df[u"text"].tolist():
            size += len(self._tokenize(sent))
            sents.append(sent)
            if size >= length:
                break
                    
        psents = [self._pretty_sent(sent)
                  for sent in sents]
        return textwrap.fill(u"\n".join(psents))

    def __unicode__(self, length=200):
        return self._make_pretty(length=length)

    def __str__(self):
        return unicode(self).encode(u"utf-8")



class Document(object):
    def __init__(self, name, text):
        self.name = name
        if isinstance(self.name, str):
            self.name = self.name.decode(u"utf-8")
        self.text = text
        if isinstance(self.text, str):
            self.text = self.text.decode(u"utf-8")

    def __str__(self):
        return unicode(self).encode(u"utf-8")
    
    def __unicode__(self):
        return self.name + u"\n" + self.text

class DocSet(object):
    def __init__(self, docs):
        self.docs = docs
