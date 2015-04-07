
class GreedyReranker:
    def greedyrerank(input_df, ranker, budget):
        summary = []
        sent_remaining = range(0, len(input_df['text']))
        length = 0
        value = 0
        while len(sent_remaining) > 0:
            max_gain = None
            best_sent = ""
            best_index = None
            for i in range(0, len(sent_remaining)):
                index = sent_remaining[i]
                combined = summary + input_df['text'][index]
                gain = (ranker(combined) - ranker(summary)) / (len(input_df['text'][index])^r)
                if gain > max_gain:
                    gain = max_gain
                    best_sent = input_df['text'][index]
                    best_index = index
             if max_gain > 0 and length + len(best_sent) <= budget:
                 summary.append(best_sent)
             sent_remaining.remove(best_index)
        max_sing_value = None
        max_sing_index = None
        max_sing_sent = ""
        for i in range(0, len(input_df['text'])):
            if ranker(input_df['text'][i]) > max_sing_value:
                max_sing_value = ranker(input_df['text'][i])
                max_sing_index = i
                max_sing_sent = input_df['text'][i]
        if max_sing_value > ranker(summary):
            return max_sing_sent
        else:
            return summary
