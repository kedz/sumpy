from sumpy.preprocessor import CorpusTfidfMixin
import copy
import numpy as np

class fmmr(CorpusTfidfMixin):
    def __init__(self, input_df):
        tfidfer = self.build_tfidf_vectorizer()
        self.tfidf_mat = tfidfer(input_df[u"words"].tolist())

    def get_fmmr_value(self, in_summary, in_indicies, not_summary,
        not_indicies, l=4):
        sum_sim = 0
        sum_red= 0
        for i in range(0, len(in_indicies)):
            for j in range(i+1, len(in_indicies)):
                first_index = in_indicies[i]
                second_index = in_indicies[j]
                sum_red += self.tfidf_mat[first_index][second_index]
        for i in range(0, len(in_indicies)):
            for j in range(i+1, len(not_indicies)):
                first_index = in_indicies[i]
                second_index = not_indicies[j]
                sum_sim += self.tfidf_mat[first_index, second_index]
        return sum_sim - l*sum_red

class GreedyReranker(fmmr):
    def greedyrerank(self, input_df, budget=200, ranker=None,r=0.3):
        #Set default ranker to fmmr if not defined
        if not ranker:
            func = fmmr(input_df)
            ranker= func.get_fmmr_value
        #initialize values
        summary = None
        summary_indicies = []
        sent_remaining = range(0, len(input_df['text']))
        length = 0
        #Keep adding sentences while there are more sentences and underbudget
        while len(sent_remaining) > 0 and length < budget:
            max_gain = None
            best_index = None
            for i in range(0, len(sent_remaining)):
                index = sent_remaining[i]
                #Get the sentences/indicies not in the summary
                remaining = input_df.copy(deep=True).drop(summary_indicies)
                remaining_indicies = range(0, len(input_df.index))
                for summary_index in summary_indicies:
                    remaining_indicies.remove(summary_index)
                #Calculate current value of summary
                old_value = ranker(summary, summary_indicies, remaining, 
                    remaining_indicies)
                #Move the current index from the remaining sent/indicies
                #to the summary sent/indicies
                remaining.drop(index)
                remaining_indicies.remove(index)
                copy_summary = None
                if summary:
                    copy_summary = summary.copy(deep=True).concat(
                        input_df.loc[[index]])
                else:
                    copy_summary = input_df.loc[[index]]
                new_value = ranker(copy_summary, summary_indicies + [index], 
                    remaining, remaining_indicies)
                #Calculate the gain and update if greater than current max
                gain = (new_value - old_value) / \
                    (len(input_df['text'][index])**r)
                if gain > max_gain:
                    gain = max_gain
                    best_index = index
            #Make sure max is pos and length under budget
            #If so, add it to the summary
            if max_gain > 0 and length + len(best_sent) <= budget:
                summary.concat(input_df.loc[[best_index]])
                summary_indicies.append(best_index)
                length = length + len(best_sent)
            sent_remaining.remove(best_index)
        #Find the max ranking value for one sentence
        max_sing_value = None
        max_sing_index = None
        max_sing_sent = ""
        for i in range(0, len(input_df['text'])):
            in_summary = input_df.loc[[i]]
            in_indicies = [i]
            out_summary = input_df.copy(deep=True).drop(i)
            out_indicies = range(0, len(input_df.index))
            out_indicies.remove(i)
            current_value = ranker(in_summary, in_indicies, out_summary, out_indicies)
            if current_value > max_sing_value:
                max_sing_value = current_value
                max_sing_index = i
                max_sing_sent = input_df['text'][i]
        #Check if the max value for one sent is greater than the summary
        remaining = input_df.copy(deep=True).drop(summary_indicies)
        remaining_indicies = range(0, len(input_df.index))
        for summary_index in summary_indicies:
            remaining_indicies.remove(summary_index)
        if max_sing_value > ranker(summary, summary_indicies, remaining, \
            remaining_indicies):
            summary =  max_sing_sent
            summary_indicies = [max_sing_index]
        #Weight the ranks so they're between 0 and 1
        ranks = np.zeros(len(input_df.index))
        for i,index in enumerate(reversed(summary_indicies)):
            ranks[index] = i / len(summary_indicies)
        input_df['rank:reranker'] = ranks
