from sumpy.annotators import WordTokenizerMixin, TfIdfCosineSimilarityMixin
import numpy as np


class SubmodularMMRMixin(TfIdfCosineSimilarityMixin):

    def build(self):
        if not hasattr(self, "lam"):
            self.lam = .3
        assert 0 <= self.lam
        
        if not hasattr(self, "scale"):
            self.scale = 1.0
        assert 0 <= self.scale

        if not hasattr(self, "budget_type"):
            self.budget_type = "word"
        assert self.budget_type in ["word", "byte"]
    
        if not hasattr(self, "budget_size"):
            self.budget_size = 400
        assert 0 < self.budget_size

        def rank(input_df, ndarray_data):
            if self.budget_type == "word":
                B = np.array(ndarray_data["RawBOWMatrix"].sum(axis=1))
                print type(B)
            elif self.budget_type == "byte":
                B = input_df["sent text"].apply(lambda x: len(x.replace("\n", ""))).values
            K = ndarray_data["TfIdfCosSimMatrix"]
            K = np.ma.masked_array(K, mask=np.diag(np.diag(K)))
            assert B.shape[0] == K.shape[0] 
            
            #B = B[[0, 25, 54, 80]]
            print B
            #K = K[[0, 25, 54, 80]][:,[0, 25, 54, 80]]
            print K
            K_S = np.ma.masked_array(K, mask=False, hardmask=False)
            print K_S
            K_V = np.ma.masked_array(K, mask=False, hardmask=False)
            print K_V
           
            print 
            print

            S = []
            B_S = 0
            V = range(K.shape[0])
            inspected_vertices = set()
            f_of_S = 0
            for rank in xrange(K.shape[0], 0, -1):
                #print "K_S"
                #print K_S
                #print "S"
                #print S
                #print "V"
                #print V
                max_gain = float("-inf")
                max_idx = None
                max_v = None
                max_f_of_S_plus_v = None
                for i, v in enumerate(V):
                    if v in inspected_vertices:
                        continue
                    S_tmp = S + [v]
                    V_tmp = V[:i] + V[i+1:]
                    #print S_tmp
                    #print V_tmp
                    #print K[S_tmp][:, V_tmp]
                    #print K[S_tmp][:, S_tmp].filled(0).sum()
                    f_of_S_plus_v = K[S_tmp][:, V_tmp].sum() - \
                        self.lam * K[S_tmp][:, S_tmp].filled(0).sum()
                    gain = (f_of_S_plus_v - f_of_S) / (B[v] ** self.scale)

                    if gain > max_gain:
                        max_gain = gain
                        max_idx = i
                        max_v = v
                        max_f_of_S_plus_v = f_of_S_plus_v
                    #print v, gain

                
                #del V[max_idx]

                if max_gain > 0 and B_S + B[max_v] <= self.budget_size:
                    print "Adding", max_v, "f(S + v) =", max_f_of_S_plus_v
                    S += [max_v]
                    del V[max_idx]
                    f_of_S = max_f_of_S_plus_v
                    print "B_v", B[max_v], "B_S", B_S, "B_S + B_v", B_S + B[max_v]
                    B_S += B[max_v]
                    input_df.ix[max_v, "f:submodular-mmr"] = rank

                inspected_vertices.add(max_v)
                     

                #else:
                        

                #print "Iter {} f(S) = {}".format(rank, f_of_S)
                #print
                #print
                #f_cut = K.sum(axis=1)
                #print f_cut
                #if rank == K.shape[0] - 2:
                #    break    

            return input_df, ndarray_data
        self._submodular_mmr = rank 

    def process(self, input_df, ndarray_data):
        return self._submodular_mmr(input_df, ndarray_data)

    def requires(self):
        return ["sent text"]
    
    def ndarray_requires(self):
        return ["TfIdfCosSimMatrix", "RawBOWMatrix"]

    def returns(self):
        return ["f:submodular-mmr"]

    def ndarray_returns(self):
        return []

    def name(self):
        return "SubmodularMMRMixin"

class MonotoneSubmodularMixin(WordTokenizerMixin):
    def build(self):
        if not hasattr(self, "k"):
            self.k = 5
        assert self.k > 0
        
        if not hasattr(self, "f_of_A") or self.f_of_A is None:
            def f_of_A(system, A, V_min_A, e, input_df, ndarray_input):
                return len(
                    set([word for words in input_df.ix[A, "words"].tolist() for word in words]))
            self.f_of_A = f_of_A

    def process(self, input_df, ndarray_data):

        input_size = len(input_df)
        S = []
        V_min_S = [i for i in xrange(input_size)]
        f_of_S = 0        
        for i in xrange(self.k):
            arg_max = None
            gain_max = 0
            f_of_S_max = 0
            for pos, elem in enumerate(V_min_S):
                S_plus_e = S + [elem]
                V_min_S_plus_e = V_min_S[:pos] + V_min_S[pos+1:]
                score = self.f_of_A(
                    self, S_plus_e, V_min_S_plus_e, elem, input_df, ndarray_data) 
                gain = score - f_of_S

                if gain > gain_max: 
                    arg_max = pos
                    gain_max = gain
                    f_of_S_max = score

            if arg_max is not None:
                S += [V_min_S[arg_max]]
                f_of_S = f_of_S_max
                del V_min_S[arg_max]

        input_df.ix[S, "f:monotone-submod"] = 1        
        input_df.ix[V_min_S, "f:monotone-submod"] = 0        
    
        return input_df, ndarray_data 

    def process2(self, input_df, ndarray_data):
        
        input_size = len(input_df)
        S = []
        N = set()

        n_of_e = input_df["nuggets"].tolist()
        V_min_S = [i for i in xrange(input_size)]
        f_of_S = 0        


        for i in xrange(self.k):
            arg_max = None
            gain_max = 0
            for pos, elem in enumerate(V_min_S):
                #print "elem", elem
                #print "S", S
                #print "V_min_S", V_min_S
                #print "n(e) =", n_of_e[elem]
                n_of_S_U_e = N.union(n_of_e[elem])
                #print "S U {e}", S + [elem]
                #print "n(S U {e})", n_of_S_U_e

                gain = self._f_of_S(n_of_S_U_e) - f_of_S
                #print "gain", gain
                #print
                if gain > gain_max: 
                    arg_max = pos
                    gain_max = gain

            if arg_max is not None:
                S = S + [V_min_S[arg_max]]
                N = N.union(n_of_e[V_min_S[arg_max]])
                f_of_S = len(N)
                
                print "ARG MAX", V_min_S[arg_max]
                print "S", S
                print "N", N
                print "f(S)", f_of_S
                
                del V_min_S[arg_max]


        print S
        print input_df
        print input_size
        input_df.ix[S, "f:monotone-submod"] = 1        
        input_df.ix[V_min_S, "f:monotone-submod"] = 0        


        return input_df, ndarray_data


    def requires(self):
        return ["words"]
    
    def ndarray_requires(self):
        return []

    def returns(self):
        return ["f:montone-submod"]

    def ndarray_returns(self):
        return []

    def name(self):
        return "MonotoneSubmod" 



