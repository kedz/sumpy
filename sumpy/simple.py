import sumpy

def lede(inputs):
    s = sumpy.system.LedeSummarizer()
    return s.summarize(inputs)

def centroid(inputs):
    s = sumpy.system.CentroidSummarizer()
    return s.summarize(inputs)

def mmr(inputs):
    s = sumpy.system.MMRSummarizer()
    return s.summarize(inputs)

def textrank(inputs):
    s = sumpy.system.TextRankSummarizer()
    return s.summarize(inputs)

def lexrank(inputs):
    s = sumpy.system.LexRankSummarizer()
    return s.summarize(inputs)
