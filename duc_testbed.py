import argparse
import pandas as pd
import os
import sumpy
import sumpy.eval

def load_docsets(duc_dir):

    docset_paths = [os.path.join(duc_dir, fname)
                    for fname in os.listdir(duc_dir)]
    docset_paths = [path for path in docset_paths if os.path.isdir(path)]
    docsets = {}
    for docset_path in docset_paths:
        docset_id, docs, models = load_docset(docset_path)
        docsets[docset_id] = {u"docs": docs, u"models": models}
    return docsets

def load_docset(docset_path):
    docset_id = os.path.split(docset_path)[1]
    docs_path = os.path.join(docset_path, u"docs")
    docs = sumpy.io.load_duc_docset(docs_path)
    models = []
    for fname in os.listdir(docset_path):
        if docset_id in fname:
            model_paths = [os.path.join(docset_path, fname, length)
                           for length in [u"200", u"400"]]
            model_sums = sumpy.io.load_duc_abstractive_summaries(model_paths)
            models.extend(model_sums)
    return docset_id, docs, models


def generate_summaries(systems, docsets):
    rouge = sumpy.eval.ROUGE(max_ngrams=2, limit=100, limit_type=u"word")
    results = []
    for docset_id in docsets.keys():
        #print docset_id
        docs = docsets[docset_id][u"docs"]
        models = docsets[docset_id][u"models"]
        sys_sums = [(system_name, unicode(sum_func(docs)))
                    for system_name, sum_func in systems]
        df = rouge.evaluate(sys_sums, models)
        results.append(df)
    return pd.concat(results).groupby(level=0).mean()

def main(duc_dir):
    print u"Loading DUC document sets from:", duc_dir 
    docsets = load_docsets(duc_dir)
    
    lede = lambda x: sumpy.lede(x)
    centroid = lambda x: sumpy.centroid(x)
    lexrank = lambda x: sumpy.lexrank(x)
    systems = [(u"lede", lede), (u"centroid", centroid),
               (u"lexrank", lexrank)]
    print generate_summaries(systems, docsets)

if __name__ == u"__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(u"-d", u"--duc-dir", required=True, type=unicode,
                        help=u"path to DUC document set directory")
    args = parser.parse_args()
    duc_dir = args.duc_dir
    main(duc_dir)

