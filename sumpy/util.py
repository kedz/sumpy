import os
import shutil
import tarfile
import re 
from datetime import datetime
import corenlp as cnlp
import json
import pkg_resources

class DUCHelper(object):

    def __init__(self, duc_path=None, sumpy_data_path=None):
        if duc_path is None:
            duc_path = os.getenv("DUC_DATA", "~/DUC")
        self.duc_path = duc_path
        if sumpy_data_path is None:
            self.sumpy_data_path = os.getenv("SUMPY_DATA",
                os.path.join(
                    os.path.expanduser("~"), ".sumpy"))
        
    def docset_iter(self, year, task):

        if year == 2003:
            if task == 2:
                duc_json_path = pkg_resources.resource_filename(
                    "sumpy", os.path.join("data", "duc03_task2.json"))
                with open(duc_json_path, "r") as f:
                    docsets = json.load(f, strict=False)

                docset_ids = sorted(docsets.keys())
                for docset_id in docset_ids:
                    ds = DUCDocset(
                        docset_id, 2003, 2,
                        docsets[docset_id]["inputs"],
                        os.path.join(
                            self.sumpy_data_path, "duc2003", "task2", 
                            docset_id, "inputs"),
                        docsets[docset_id]["models"],
                        os.path.join(
                            self.sumpy_data_path, "duc2003", "task2", 
                            docset_id, "models"))

                    yield ds


        elif year == 2004:
            if task == 2:
                duc_json_path = pkg_resources.resource_filename(
                    "sumpy", os.path.join("data", "duc04_task2.json"))
                with open(duc_json_path, "r") as f:
                    docsets = json.load(f, strict=False)

                docset_ids = sorted(docsets.keys())
                for docset_id in docset_ids:
                    ds = DUCDocset(
                        docset_id, 2004, 2,
                        docsets[docset_id]["inputs"],
                        os.path.join(
                            self.sumpy_data_path, "duc2004", "task2", 
                            docset_id, "inputs"),
                        docsets[docset_id]["models"],
                        os.path.join(
                            self.sumpy_data_path, "duc2004", "task2", 
                            docset_id, "models"))

                    yield ds

#        elif year == 2007:
#            if task == 2:
#                for docset_id in self.duc07_task2_docset_ids:
#                    dsA = DUCDocset(
#                        docset_id, 2007, 2,
#                        self.duc07_task2[docset_id]["A"]["inputs"],
#                        os.path.join(self.duc07_task2_docsets_path, 
#                            "{}-A".format(docset_id)),
#                        self.duc07_task2[docset_id]["A"]["models"],
#                        os.path.join(self.duc07_task2_models_path))
#                    dsB = DUCDocset(
#                        docset_id, 2007, 2,
#                        self.duc07_task2[docset_id]["B"]["inputs"],
#                        os.path.join(self.duc07_task2_docsets_path, 
#                            "{}-B".format(docset_id)),
#                        self.duc07_task2[docset_id]["B"]["models"],
#                        os.path.join(self.duc07_task2_models_path))
#                    dsC = DUCDocset(
#                        docset_id, 2007, 2,
#                        self.duc07_task2[docset_id]["C"]["inputs"],
#                        os.path.join(self.duc07_task2_docsets_path, 
#                            "{}-C".format(docset_id)),
#                        self.duc07_task2[docset_id]["C"]["models"],
#                        os.path.join(self.duc07_task2_models_path))
#
#                    ds = DUCUpdateDocset(
#                        docset_id, year, task, [dsA, dsB, dsC])
#                    yield ds

        else:
            raise Exception("Bad argument: year is {}".format(year))

    def docsets(self, year, task):
        if year == 2003:
            if task == 2:
                return DUCDocsets([ds for ds in self.docset_iter(2003, 2)])
            else:
                raise Exception("Bad argument: task is {}".format(task))
        elif year == 2004:
            if task == 2:
                return DUCDocsets([ds for ds in self.docset_iter(2004, 2)])
            else:
                raise Exception("Bad argument: task is {}".format(task))
        else:
            raise Exception("Bad argument: year is {}".format(year))

    def install(self, year, task):
        if year == 2001:
            raise Exception("Not implemented!")
        elif year == 2002:
            raise Exception("Not Implemented!")
        elif year == 2003:
            self._install_duc03_task2()
        elif year == 2004:
            self._install_duc04_task2()
        else:
            raise Exception("Not implemented!")

    def _install_duc03_task2(self):
        data_path = os.path.join(self.sumpy_data_path, "duc2003", "task2")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        data_path_duc = os.path.join(
            self.duc_path, "DUC2003_Summarization_Documents.tgz") 
        data_path_models = os.path.join(
            self.duc_path, "detagged.duc2003.abstracts.tar.gz")

        if not os.path.exists(data_path_duc):
            raise Exception("{} does not exist. " \
                            "Please obtain this file from NIST.".format(
                                data_path_duc))
        if not os.path.exists(data_path_models):
            raise Exception("{} does not exist. " \
                            "Please obtain this file from NIST.".format(
                                data_path_models))
        

        docsets = {}
        
        docs_tar = os.path.join("DUC2003_Summarization_Documents", 
            "duc2003_testdata", "task2", "task2.docs.tar.gz")
        with tarfile.open(name=data_path_duc, mode="r") as tf:
            for m in tf.getmembers():
                if m.name == docs_tar:
                    break

            f = tf.extractfile(m)
            from StringIO import StringIO
            b = StringIO(f.read())
            with tarfile.open(fileobj=b, mode="r") as dtf:
                for m in dtf.getmembers():
                    path, doc_id = os.path.split(m.name)
                    _, docset_id = os.path.split(path)
                    text = dtf.extractfile(m).read()
                    docset_id = docset_id.upper()[:-1]
                    docset = docsets.get(
                        docset_id, {"inputs": [], "models": []}) 
                    docset["inputs"].append({"input id": doc_id, "text": text})
                    docsets[docset_id] = docset
        with tarfile.open(name=data_path_models, mode="r") as tf:
            for m in tf.getmembers():
                path, model = os.path.split(m.name)
                if os.path.split(path)[1] == "models":
                    if re.search(r'D\d{5}\.\w\.100\.\w\.\w.html', model):
                        docset_id = model.split(".")[0]
                        model_id = os.path.splitext(model)[0]
                        text = tf.extractfile(m).read()
                        docsets[docset_id]["models"].append(
                            {"model id": model_id,
                             "text": text})

        #annotators=["tokenize", "ssplit"]
        #with cnlp.Server(annotators=annotators) as pipeline:
        for docset_id, docset in docsets.items():
            inputs_path = os.path.join(data_path, docset_id, "inputs")
            if not os.path.exists(inputs_path):
                os.makedirs(inputs_path)
            for input in docset["inputs"]:
                input_path = os.path.join(inputs_path, input["input id"])
                with open(input_path, "wb") as f:
                    f.write(input["text"])
   
            models_path = os.path.join(data_path, docset_id, "models")
            if not os.path.exists(models_path):
                os.makedirs(models_path)
            for model in docset["models"]:
                model_path = os.path.join(models_path, model["model id"])
                #doc = pipeline.annotate(model["text"])

                with open(model_path, "wb") as f:
                    f.write(model["text"])
                    #for sent in doc:
                    #    line = " ".join([str(tok) for tok in sent]) + "\n"
                    #    f.write(line)


    def _install_duc04_task2(self):
        data_path = os.path.join(self.sumpy_data_path, "duc2004", "task2")
        if not os.path.exists(data_path):
            os.makedirs(data_path)
        data_path_duc = os.path.join(
            self.duc_path, "DUC2004_Summarization_Documents.tgz") 
        data_path_models = os.path.join(
            self.duc_path, "duc2004_results.tgz")

        if not os.path.exists(data_path_duc):
            raise Exception("{} does not exist. " \
                            "Please obtain this file from NIST.".format(
                                data_path_duc))
        if not os.path.exists(data_path_models):
            raise Exception("{} does not exist. " \
                            "Please obtain this file from NIST.".format(
                                data_path_models))
        
        docsets = {}
        tgt_path = os.path.join("DUC2004_Summarization_Documents",
            "duc2004_testdata", "tasks1and2", "duc2004_tasks1and2_docs", 
            "docs")
        with tarfile.open(name=data_path_duc, mode="r") as tf:
            for m in tf.getmembers():
                path, doc_id = os.path.split(m.name)
                path, docset_id = os.path.split(path)
                if path == tgt_path:
                    docset_id = docset_id.upper()[:-1]
                    text = tf.extractfile(m).read() 
                    docset = docsets.get(
                        docset_id, {"inputs": [], "models": []}) 
                    docset["inputs"].append({"input id": doc_id, "text": text})
                    docsets[docset_id] = docset
        tgt_path = os.path.join("duc2004_results", "ROUGE", 
            "duc2004.task2.ROUGE.models.tar.gz")
        with tarfile.open(name=data_path_models, mode="r") as tf:
            for m in tf.getmembers():
                if m.name == tgt_path:
                    break
            models_tar = tf.extractfile(m)
            with tarfile.open(fileobj=models_tar, mode="r") as mtf:
                for m in mtf.getmembers():
                    model_id = os.path.split(m.name)[1]
                    docset_id = model_id.split(".")[0]
                    text = mtf.extractfile(m).read()
                    docsets[docset_id]["models"].append(
                        {"model id": model_id,
                         "text": text})

        #annotators=["tokenize", "ssplit"]
        #with cnlp.Server(annotators=annotators) as pipeline:
        for docset_id, docset in docsets.items():
            inputs_path = os.path.join(data_path, docset_id, "inputs")
            if not os.path.exists(inputs_path):
                os.makedirs(inputs_path)
            for input in docset["inputs"]:
                input_path = os.path.join(inputs_path, input["input id"])
                with open(input_path, "wb") as f:
                    f.write(input["text"])
   
            models_path = os.path.join(data_path, docset_id, "models")
            if not os.path.exists(models_path):
                os.makedirs(models_path)
            for model in docset["models"]:
                model_path = os.path.join(models_path, model["model id"])
                
                #doc = pipeline.annotate(model["text"])

                with open(model_path, "wb") as f:
                    f.write(model["text"])
                    #for sent in doc:
                    #    line = " ".join([str(tok) for tok in sent]) + "\n"
                    #    f.write(line)


        

#    def _install_duc01_task2(self):
#
#        data_path = os.path.join(self.sumpy_data_path, "duc2001", "task2")
#        if not os.path.exists(data_path):
#            os.makedirs(data_path)
#        data_path_duc = os.path.join(
#            self.duc_path, "DUC2001_Summarization_Documents.tgz") 
#        
#        if not os.path.exists(data_path_duc):
#            raise Exception("{} does not exist. " \
#                            "Please obtain this file from NIST.".format(
#                                data_path_duc))
#                            
#        docments_tar_path = os.path.join("DUC2001_Summarization_Documents", 
#            "data", "testtraining", "Duc2001testtraining.tar.gz")
#
#        with tarfile.open(name=data_path_duc, mode="r") as tf:
#            mem_documents_tar = [m for m in tf.getmembers()
#                                 if m.name == docments_tar_path]
#            tf.extractall(members=mem_documents_tar) 
#        documents_tar_path = os.path.join(
#            "DUC2001_Summarization_Documents", "data", "testtraining",
#            "Duc2001testtraining.tar.gz")
#
#        if not os.path.exists(documents_tar_path):
#            raise Exception("Failed to extract DUC 2001 documents!")
#
#        with tarfile.open(docments_tar_path, mode="r") as tf:
#            tf.extractall()
#
#        documents_path = "duc2002testtraining"
#        if not os.path.exists(documents_path):
#            raise Exception("Failed to extract DUC 2001 documents!")
#
#        docsets = {}
#        for docset_id in os.listdir(documents_path):
#            docset_path = os.path.join(documents_path, docset_id)
#            articles = []
#            for article_name in os.listdir(docset_path):
#                if article_name.startswith("ap"):
#                    year = 1900 + int(article_name[2:4])
#                    month = int(article_name[4:6])
#                    day = int(article_name[6:8])
#                    ts = datetime(year, month, day)
#                elif article_name.startswith("wsj"):
#                    year = 1900 + int(article_name[3:5])
#                    month = int(article_name[5:7])
#                    day = int(article_name[7:9])
#                    ts = datetime(year, month, day)
#                elif article_name.startswith("la"):
#                    year = 1900 + int(article_name[6:8])
#                    month = int(article_name[2:4])
#                    day = int(article_name[4:6])
#                    ts = datetime(year, month, day)
#                elif article_name.startswith("ft"):
#                    year = 1900 + int(article_name[2:4])
#                    month = int(article_name.split("-")[0][4:])
#                    ts = datetime(year, month, 1)
#                elif article_name.startswith("fbis"):
#                    ts = datetime(1977,1,1)
#                elif article_name.startswith("sjmn"):
#                    ts = datetime(91,1,1)
#                else:
#                    raise Exception("Found unsual file here {}".format(
#                        article_name))
#                print article_name, ts
#                article_path = os.path.join(
#                    docset_path, article_name, "{}.body".format(article_name))
#                with open(article_path, "rb") as f:
#                    content = f.read()
#                articles.append({"input id": article_name, 
#                                  "raw text": content,
#                                  "timestamp": ts})
#            docsets[docset_id] = articles
#
#        shutil.rmtree("DUC2001_Summarization_Documents")
#        shutil.rmtree(documents_path)

class DUCDocsets(object):
    def __init__(self, docsets):
        self._docsets = {ds.docset_id: ds for ds in docsets}

    def __getitem__(self, ds_id):
        return self._docsets[ds_id]

class DUCDocset(object):
    def __init__(self, docset_id, year, task, inputs, input_root, 
            models, model_root):
        self.docset_id = docset_id
        self.year = year
        self.task = task
        self.inputs = inputs
        self.input_root = input_root
        self.models = models
        self.model_root = model_root
    
    def __str__(self):
        return "DUCDocset({}, {}, {}, {} inputs, {}, {} models, {})".format(
            self.docset_id, self.year, self.task, len(self.inputs), 
            self.input_root[:10] + "...", len(self.models), 
            self.model_root[:10] + "...") 

    def input_iter(self):
        for doc_id in self.inputs:
            timestamp_t = int(doc_id[3:7]), int(doc_id[7:9]), int(doc_id[9:11])
            timestamp = datetime(*timestamp_t)

            yield DUCDocument(
                    doc_id, timestamp, os.path.join(self.input_root, doc_id))

    def model_iter(self):
        for doc_id in self.models:
            yield DUCModel(doc_id, os.path.join(self.model_root, doc_id))

class DUCUpdateDocset(object):
    def __init__(self, docset_id, year, task, docsets):
        self.docset_id = docset_id
        self.year = year
        self.task = task
        self.docsets = docsets
    
    def update_iter(self):
        for update_ds in self.docsets:
            yield update_ds

class DUCDocument(object):
    def __init__(self, doc_id, timestamp, path):
        self.doc_id = doc_id
        self.timestamp = timestamp
        self.path = path
        self._text = None

    def _read(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self._text = f.read()
        else:
            raise Exception("DUCDocument {} not found at path {}".format(
                self.doc_id, self.path))

    def __str__(self):
        if self._text is None:
            self._read()
        return self._text

    def __unicode__(self):
        if self._text is None:
            self._read()
        return self._text.decode("utf-8")

    def __bytes__(self):
        if self._text is None:
            self._read()
        return self._text

class DUCModel(object):
    def __init__(self, doc_id, path):
        self.doc_id = doc_id
        self.path = path
        self._text = None

    def _read(self):
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self._text = f.read()
        else:
            raise Exception("DUCModel {} not found at path {}".format(
                self.doc_id, self.path))

    def __str__(self):
        if self._text is None:
            self._read()
        return self._text

    def __unicode__(self):
        if self._text is None:
            self._read()
        return self._text.decode("utf-8")

    def __bytes__(self):
        if self._text is None:
            self._read()
        return self._text
