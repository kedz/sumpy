import pickle
import numpy as np
from itertools import combinations
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
import pkg_resources
import os
import re
import sys
from pywsd import disambiguate

class TargetEntityMixin(object):
    def targetentityrank(self, input_df, target_entity):
        counts = np.zeros(len(input_df.index))
        for i in range(0, len(input_df['ne'])):
            if target_entity in input_df['ne'][i]:
                counts[i] = 1
        input_df[u'rank:leadentity'] = counts
        
class ConceptMixin(object):
    def get_binary_concepts(self):
        return self.binary_concepts

    def get_num_model_concepts(self, model, file_name=None):
        if file_name and os.path.isfile(file_name+'.model'):
            model_file = open(file_name+'.model', 'rb')
            [self._num_model_concepts] = pickle.load(model_file)
            model_file.close()
            return
        self.m_concept_sets = {}
        self._num_model_concepts = 0
        print 'started'
        synsets = disambiguate(model)
        print 'done'
        for synset in synsets:
            if synset[1]:
                if not synset[1] in self.m_concept_sets.keys():
                    hypo = synset[1].hyponyms()
                    hyper = synset[1].hypernyms()
                    new_concept = hypo + [synset[1]]
                    new_concept.extend(hyper)
                    for c_synset in new_concept:
                        if c_synset in self.m_concept_sets.keys():
                            self.m_concept_sets[c_synset].append([new_concept])
                        else:
                            self.m_concept_sets[c_synset] = [[new_concept]]
                    self._num_model_concepts += 1
        if file_name:
            model_file = open(file_name+'.model', 'ab+')
            pickle.dump([self._num_model_concepts], model_file)
            model_file.close()
        print self._num_model_concepts
         
    def conceptrank(self, input_df, file_name=None):
        if file_name and os.path.isfile(file_name+'.concepts'):
            concept_file = open(file_name+'.concepts', 'rb')
            [self.binary_concepts, self.concept_sizes] = pickle.load(concept_file)
            concept_file.close()
            return
        #initialize counts
        self.synsets = []
        self.concept_sets = {}
        self.concept_sizes = {}
        self.binary_concepts = {}
        num_concepts = 0
        #Go through all sent
        for i, sent in enumerate(input_df['text']):
            self.synsets.append(disambiguate(sent))
            #Go through each word with synset
            for sent_synset in self.synsets[i]:
                if sent_synset[1]:
                    #If not, add a new concept
                    if not sent_synset[1] in self.concept_sets.keys():
                        #Create concept from hyper/hyponyms
                        num_concepts += 1
                        sent_hypo = sent_synset[1].hyponyms()
                        sent_hyper = sent_synset[1].hypernyms()
                        new_concept = sent_hypo + [sent_synset[1]]
                        new_concept.extend(sent_hyper)
                        for c_synset in new_concept:
                            if c_synset in self.concept_sets.keys():
                                    self.concept_sets[c_synset]\
                                    .append([new_concept])
                            else:
                                self.concept_sets[c_synset] = [[new_concept]]
        #Go through each sent's words
        for i,sent in enumerate(input_df['text']):
            self.binary_concepts[i] = {}
            for sent_synset in self.synsets[i]:
                if sent_synset[1]:
                    concepts_list = self.concept_sets[sent_synset[1]]
                    for concepts in concepts_list:
                        for concept in concepts:
                            concept = str(concept)
                            if concept in self.binary_concepts[i].keys():
                                self.binary_concepts[i][concept] += 1
                            else:
                                self.binary_concepts[i][concept] = 1
                            if concept in self.concept_sizes.keys():
                                self.concept_sizes[concept] += 1
                            else:
                                self.concept_sizes[concept] = 1
        if file_name:
            output = open(file_name + '.concepts', 'ab+')
            to_save = [self.binary_concepts, self.concept_sizes]
            pickle.dump(to_save, output)
            output.close()
            return 
        #Find rank depending on the size of the concepts
        #that the sent contains
        ranks = np.zeros(len(input_df.index))
        for i in range(0, len(input_df['text'])):
            for key in self.binary_concepts[i].keys():
                    ranks[i] += self.concept_sizes[key]
        ranks = (ranks * 1.0) / np.amax(ranks)
        input_df[u'rank:concept'] = ranks

class LocationMixin(object):
    def locationrank(self, input_df):
        ranks = input_df['doc position']
        ranks = [1 - (rank * 1.0 / np.amax(ranks)) for rank in ranks]
        input_df[u'rank:location'] = ranks

class CountPronounsMixin(object):
    def countpronounsrank(self, input_df):
        counts = np.zeros(len(input_df.index))
        for i in range(0, len(input_df['pos'])):
            sent = input_df['pos'][i]
            count = 0
            for pos in sent:
                if pos[:2] == 'PR':
                    count = count + 1
            counts[i] = count
        counts = counts * 1.0 / np.amax(counts)
        counts = 1 - counts
        input_df[u'rank:countpronoun'] = counts

class SentLengthMixin(object):
    def sentlengthrank(self, input_df):
        lengths = np.zeros(len(input_df.index))
        for i in range(0, len(input_df['pos'])):
            length = 0
            for j in range(0, len(input_df['pos'][i])):
                pos = input_df['pos'][i][j]
                if re.match("^[A-Za-z]*$", pos) and not(pos == 'CD'): 
                    length = length + 1
            if length > 30:
                length = length - 30
            elif length < 15:
                length = 15 - length
            else: 
                length = 0
            lengths[i] = length
        lengths = lengths * 1.0 / np.amax(lengths)
        lengths = 1 - lengths
        input_df[u'rank:sentlength'] = lengths

class LeadValuesMixin(object):
    def leadvaluesrank(self, input_df):
        lead_path =  pkg_resources.resource_filename("sumpy", 
                        os.path.join("data", "lead_words.txt"))
        self._leadwords = []
        with open(lead_path, u"r") as f:
            text = f.readlines()
            for line in text:
                line_split = line.split()
                self._leadwords.append(line_split[0])
        lead_score = np.zeros(len(input_df.index))
        for i in range(0, len(input_df.index)):
            lead_score[i] = self._get_lead_values(input_df['lem'][i])
        lead_score = lead_score / np.amax(lead_score)
        input_df[u'rank:leadvalue'] = lead_score

    def _get_lead_values(self, sent):
        word_count = 0
        lead_word_count = 0
        for token in sent:
            word_count = word_count + 1
            if token in self._leadwords:
                lead_word_count = lead_word_count + 1
        return float(lead_word_count) / word_count

class DiscourseMarkersMixin(object):
    def discoursemarkersrank(self, input_df):
        markers_path = pkg_resources.resource_filename("sumpy",
            os.path.join("data", "explicit_discourse_markers.txt"))
        self._dismarkers = []
        with open(markers_path, u'r') as f:
            self._dismarkers = f.readlines()
        for i in range(0, len(self._dismarkers)):
            self._dismarkers[i] = self._dismarkers[i].strip().lower()
        dismarker_score = np.ones(len(input_df.index))
        for i in range(0, len(input_df.index)):
            text = input_df['text'][i].strip().lower()
            text = ''.join(filter(str.isalpha, text))
            for dismarker in self._dismarkers:
                if text.find(dismarker) == 0:
                    dismarker_score[i] = 0
                    break
        input_df[u'rank:dismarker'] = dismarker_score

class VerbSpecificityMixin(object):
    def verbspecificityrank(self, input_df):
        verb_path = pkg_resources.resource_filename("sumpy", 
                        os.path.join("data", "verb_specificity.txt"))
        self._verbspec = {}
        with open(verb_path, u"r") as f:
            text = f.readlines()
            for line in text:
                line_split = line.split()
                self._verbspec[line_split[0]] = line_split[1]
        verb_score = np.zeros(len(input_df.index))
        for i in range(0, len(input_df.index)):
            verb_score[i] = self._get_verb_specificity(input_df['lem'][i], input_df['pos'][i])
        verb_score = verb_score / np.amax(verb_score)
        input_df[u'rank:verbspec'] = verb_score

    def _get_verb_specificity(self, sent, pos):
        max_val = 0
        for i, token in enumerate(sent):
            if pos[i][:2] == 'VB' and token in self._verbspec.keys() and float(self._verbspec[token]) > max_val:
                max_val = float(self._verbspec[token])
        return max_val

class DEMSRankerMixin(LeadValuesMixin, VerbSpecificityMixin,
                      CountPronounsMixin, SentLengthMixin, 
                      LocationMixin, ConceptMixin, TargetEntityMixin,
                      DiscourseMarkersMixin):
    def demsrank(self, input_df, target_entity, lead_word_weight=1, verb_spec_weight=1,
                count_pronoun_weight=1, sent_length_weight=1,
                location_weight=1, concept_weight=1, lead_entity_weight=1,
                dismarkers_weight=1,):
        self.leadvaluesrank(input_df)
        self.verbspecificityrank(input_df)
        self.countpronounsrank(input_df)
        self.sentlengthrank(input_df)
        self.locationrank(input_df)
        self.conceptrank(input_df)
        self.targetentityrank(input_df, target_entity)
        self.discoursemarkersrank(input_df)
        input_df[u"rank:demsrank"] = lead_word_weight * input_df[u'rank:leadvalue'] \
            + verb_spec_weight * input_df[u'rank:verbspec'] \
            + count_pronoun_weight * input_df[u'rank:countpronoun'] \
            + sent_length_weight * input_df[u'rank:sentlength'] \
            + location_weight * input_df[u'rank:location'] \
            + concept_weight * input_df[u'rank:concept'] \
            + lead_entity_weight * input_df[u'rank:leadentity'] \
            + dismarkers_weight * input_df[u'rank:dismarker']

class LedeRankerMixin(object):
    def rank_by_lede(self, input_df):
        input_df[u"rank:lede"] = 0
        input_df.loc[input_df[u"doc position"] == 1, u"rank:lede"] = 1

class TextRankMixin(object):
    def textrank(self, input_df, directed=u"undirected", d=0.85, 
                 max_iters=20, tol=.0001):
        word_sets = [set(words) for words in input_df[u"words"].tolist()]
        max_sents = len(word_sets)
        K = self.compute_kernel(word_sets)
        M_hat = (d * K) + \
                (float(1 - d) / max_sents) * np.ones((max_sents, max_sents))
        M_hat /=  np.sum(M_hat, axis=0)
        r = np.ones((max_sents), dtype=np.float64) / max_sents

        converged = False
        for n_iter in xrange(max_iters):
            last_r = r
            r = np.dot(M_hat, r)

            if (np.abs(r - last_r) < tol).any():
                converged = True
                break

        if not converged:
            print "warning:", 
            print "textrank failed to converged after {} iters".format(
                max_iters)
        input_df["rank:textrank"] = r

    def compute_kernel(self, word_sets, directed=u"undirected"):
        """Compute similarity matrix K ala text rank paper. Should this be
        a ufunc???"""
        n_sents = len(word_sets)
        M = np.zeros((n_sents, n_sents), dtype=np.float64)
        for i, j in combinations(xrange(n_sents), 2):
            s_i = word_sets[i]
            s_j = word_sets[j] 
            val = len(s_i.intersection(s_j))
            val /= np.log(len(s_i) * len(s_j))
            M[i,j] = val
            M[j,i] = val
        return M

class LexRankMixin(object):
    def lexrank(self, input_df, tfidf_mat, d=.85, max_iters=20, tol=.0001):
        max_sents = len(input_df)
        K = cosine_similarity(tfidf_mat)        
        M_hat = (d * K) + \
                (float(1 - d) / max_sents) * np.ones((max_sents, max_sents))
        M_hat /=  np.sum(M_hat, axis=0)
        r = np.ones((max_sents), dtype=np.float64) / max_sents

        converged = False
        for n_iter in xrange(max_iters):
            last_r = r
            r = np.dot(M_hat, r)

            if (np.abs(r - last_r) < tol).any():
                converged = True
                break

        if not converged:
            print "warning:", 
            print "lexrank failed to converged after {} iters".format(
                max_iters)
        input_df["rank:lexrank"] = r


class CentroidScoreMixin(object):
    def centroid_score(self, input_df, tfidf_mat):
        centroid = tfidf_mat.sum(axis=0)
        assert centroid.shape[1] == tfidf_mat.shape[1]
        indices = tfidf_mat.indices
        indptr = tfidf_mat.indptr
        nnz = tfidf_mat.nnz
        occurence_mat = csr_matrix(
            (np.ones((nnz)), indices, indptr), shape=tfidf_mat.shape)
        input_df[u"rank:centroid_score"] = occurence_mat.dot(centroid.T)


