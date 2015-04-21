import sumpy

def lede(inputs):
    s = sumpy.system.LedeSummarizer()
    return s.summarize(inputs)

def textrank(inputs):
    s = sumpy.system.TextRankSummarizer()
    return s.summarize(inputs)

def lexrank(inputs):
    s = sumpy.system.LexRankSummarizer()
    return s.summarize(inputs)

def centroid(inputs):
    s = sumpy.system.CentroidSummarizer()
    return s.summarize(inputs)

def dems(inputs):
    s = sumpy.system.DEMSSummarizer()
    return s.summarize(inputs)

def reranker(inputs):
    s = sumpy.system.RerankerSummarizer()
    return s.summarize(inputs)
