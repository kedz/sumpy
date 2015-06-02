from sumpy.annotators import TfIdfCosineSimilarityMixin
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


