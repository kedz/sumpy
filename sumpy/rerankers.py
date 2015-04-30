from sumpy.preprocessor import CorpusTfidfMixin
import copy
import numpy as np

class fmmr(CorpusTfidfMixin):
    def __init__(self, input_df):
        tfidfer = self.build_tfidf_vectorizer()
        self.tfidf_mat = tfidfer(input_df[u"words"].tolist())
        self.input_df = input_df

    def rank(self, in_indices, not_indices, l=4):
        sum_sim = 0
        sum_red= 0
        for i in range(0, len(in_indices)):
            for j in range(i+1, len(in_indices)):
                first_index = in_indices[i]
                second_index = in_indices[j]
                sum_red += self.tfidf_mat[first_index][second_index]
        for i in range(0, len(in_indices)):
            for j in range(i+1, len(not_indices)):
                first_index = in_indices[i]
                second_index = not_indices[j]
                sum_sim += self.tfidf_mat[first_index, second_index]
        return sum_sim - l*sum_red

class GreedyReranker(fmmr):
    def greedyrerank(self, input_df, budget=200, ranker=None,r=0.3):
        #Set default ranker to fmmr if not defined
        if not ranker:
            func = fmmr(input_df)
            ranker= func.rank
        #initialize values
        summary_indices = []
        sent_remaining = range(0, len(input_df['text']))
        length = 0
        #Keep adding sentences while there are more sentences and underbudget
        while len(sent_remaining) > 0 and length < budget:
            max_gain = None
            best_index = None
            for i in range(0, len(sent_remaining)):
                index = sent_remaining[i]
                #Get the sentences/indices not in the summary
                remaining_indices = range(0, len(input_df.index))
                for summary_index in summary_indices:
                    remaining_indices.remove(summary_index)
                #Calculate current value of summary
                old_value = ranker(summary_indices, 
                    remaining_indices)
                #Move the current index from the remaining sent/indices
                #to the summary sent/indices
                remaining_indices.remove(index)
                new_value = ranker(summary_indices + [index], 
                    remaining_indices)
                #Calculate the gain and update if greater than current max
                gain = (new_value - old_value) / \
                    (len(input_df['text'][index])**r)
                if gain > max_gain:
                    gain = max_gain
                    best_index = index
            #Make sure max is pos and length under budget
            #If so, add it to the summary
            if max_gain > 0 and length + len(best_sent) <= budget:
                summary_indices.append(best_index)
                length = length + len(best_sent)
            sent_remaining.remove(best_index)
        #Find the max ranking value for one sentence
        max_sing_value = None
        max_sing_index = None
        for i in range(0, len(input_df['text'])):
            in_indices = [i]
            out_indices = range(0, len(input_df.index))
            out_indices.remove(i)
            current_value = ranker(in_indices, out_indices)
            if current_value > max_sing_value:
                max_sing_value = current_value
                max_sing_index = i
        #Check if the max value for one sent is greater than the summary
        remaining_indices = range(0, len(input_df.index))
        for summary_index in summary_indices:
            remaining_indices.remove(summary_index)
        current_score = ranker(summary_indices, remaining_indices)
        if max_sing_value > current_score:
            summary_indices = [max_sing_index]
        self.indices = summary_indices
        #Weight the ranks so they're between 0 and 1
        ranks = np.zeros(len(input_df.index))
        for i,index in enumerate(reversed(summary_indices)):
            ranks[index] = i / len(summary_indices)
        input_df['rank:reranker'] = ranks
