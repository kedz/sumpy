import os
import re

def load_duc_docset(input_source):
    docs = DucSgmlReader().read(input_source)
    return docs

def load_duc_abstractive_summaries(input_source):
    models = DucAbstractSgmlReader().read(input_source)
    return models

class FileInput(object):

    def gather_paths(self, source):
        """Determines the type of source and return an iterator over input 
        document paths. If source is a str or unicode 
        object, determine if it is also a directory and return an iterator
        for all directory files; otherwise treat as a single document input.
        If source is any other iterable, treat as an iterable of file 
        paths."""

        if isinstance(source, str) or isinstance(source, unicode):
            if os.path.isdir(source):
                paths = [os.path.join(source, fname) 
                         for fname in os.listdir(source)]
                for path in paths:
                    yield path
            else:
                yield source
        
        else:
            try:
                for path in source:
                    yield path
            except TypeError:
                print source, 'is not iterable'

class DucSgmlReader(FileInput):

    def read(self, input_source):
        docs = []
        for path in self.gather_paths(input_source):
            with open(path, u"r") as f:
                sgml = "".join(f.readlines())
                m = re.search(r"<TEXT>(.*?)</TEXT>", sgml, flags=re.DOTALL)
                if m is None:
                    raise Exception("TEXT not found in " + path)
                text = m.group(1).strip()
                text_clean = re.sub(r"<[^>]*?>", r"", text)
                docs.append(text_clean)
        return docs

class DucAbstractSgmlReader(FileInput):

    def read(self, input_source):
        docs = []
        for path in self.gather_paths(input_source):
            with open(path, u"r") as f:
                sgml = "".join(f.readlines())
                m = re.search(r"<SUM[^>]+>(.*?)</SUM>", sgml, flags=re.DOTALL)
                if m is None:
                    raise Exception("SUM not found in " + path)
                text = m.group(1).strip()
                docs.append(text)
        return docs
