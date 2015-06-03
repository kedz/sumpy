import re
import textwrap

class Summary(object):
    def __init__(self, df):
        self._df = df

    def budget(self, type="byte", size=600):
        summary = []
        if size == "all":
            summary = self._df["sent text"].tolist()
        elif type == "word":
            remaining = size
            for idx, sent in self._df.iterrows():
                num_words = min(len(sent["words"]), remaining)
                summary.append(u" ".join(sent["words"][0 : num_words]))
                remaining -= num_words
                if remaining < 1:
                    break
        elif type == "byte":
            remaining = size
            for idx, sent in self._df.iterrows():
                num_chars = min(len(sent["sent text"]), remaining)
                print num_chars
                summary.append(sent["sent text"][0 : num_chars])
                remaining -= num_chars
                if remaining < 1:
                    break
        return u"\n".join(textwrap.fill(u"{}) {}".format(i, sent))
                          for i, sent in enumerate(summary, 1)) + u" ..." 

    def __unicode__(self):
        return self.budget()

    def __str__(self):
        return unicode(self).encode("utf-8")


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
