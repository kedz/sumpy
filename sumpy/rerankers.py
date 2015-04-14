from sumpy.preprocessor import CorpusTfidfMixin

class fmmr:
    def __init__(self, tfidf_mat):
        self.tfidf = tfidf_mat

    def get_fmmr_value(self, in_summary, in_indicies, not_summary,
        not_indicies, l=4):
        sum_sim = 0
        sum_red= 0
        for i in range(0, len(in_indicies)):
            for j in range(i+1, len(in_indicies)):
                first_index = in_summary[i]
                second_index = in_summary[j]
                sum_red += tfidf_mat[first_index][second_index]
        for i in range(0, len(in_indicies)):
            for j in range(i+1, len(not_indicies):
                first_index = in_summary[i]
                second_index = out_summary[j]
                sum_sim += tfidf_mat[first_index, second_index]
        return sum_sim - l*sum_red

class GreedyReranker(fmmr):
    def greedyrerank(input_df, budget, ranker=None,r=0.3):
        if not ranker:
            tfidfer = self.build_tfidf_vectorizer()
            tfidf_mat = tfidfer(input_df[u"words"]).tolist())
            func = fmmr(tfidf_mat)
            ranker= func.get_fmmr_value
        summary = None
        summary_indicies = []
        remaining = input_df.copy(deep=True)
        remaining_indicies = range(0, len(input_df.index))
        sent_remaining = range(0, len(input_df['text']))
        length = 0
        while len(sent_remaining) > 0:
            max_gain = None
            best_index = None
            for i in range(0, len(sent_remaining)):
                index = sent_remaining[i]
                copy_remaining = remaining.copy(deep=True).drop(index)
                copy_summary = None
                if summary:
                    copy_summary = summary.copy(deep=True).concat(input_df.loc[[index]])
                else:
                    copy_summary = input_df.loc[[index]]
                new_value = ranker(copy_summary, summary_indicies + [index], 
                    copy_remaining, remaining_indicies[:].remove(index)
                old_value = ranker(summary, summary_indicies, remaining, remaining_indicies)
                gain = (new_value - old_value) / (len(input_df['text'][index])^r)
                if gain > max_gain:
                    gain = max_gain
                    best_index = index
             if max_gain > 0 and length + len(best_sent) <= budget:
                 summary.concat(input_df.loc[[best_index]]
                 summary_indicies.append(best_index)
                 remaining = remaining.drop(best_index)
                 remaining_indicies.remove(best_index)
                 length = length + len(best_sent)
             sent_remaining.remove(best_index)
        max_sing_value = None
        max_sing_index = None
        max_sing_sent = ""
        for i in range(0, len(input_df['text'])):
            if ranker(input_df.loc[[i]],[i], input_df.copy(deep=True).drop(i),
                 range(0, len(input_df.index).remove(i)) > max_sing_value:
                max_sing_value = ranker(input_df['text'][i])
                max_sing_index = i
                max_sing_sent = input_df['text'][i]
        if max_sing_value > ranker(summary):
            return max_sing_sent
        else:
            return summary
