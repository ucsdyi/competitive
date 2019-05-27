import numpy as np
log = np.log
exp = np.exp

##### The improved Good-Turing estimator in the paper "Competitive Distribution Estimation: Why is Good-Turing Good" by Orlitsky & Suresh, 2015.
def Good_Turing(D_multiplicity, profile, k):
    n = 0
    Key = D_multiplicity.keys()
    D_labeled_distribution = {}
    total = 0
    Phi0 = 0
    for j in Key:
        Phi0 += len(D_multiplicity[j])
    Phi0 = k - Phi0
    for j in Key:
        symbol_list = D_multiplicity[j]
        if j<= len(profile)-1:
            if j>profile[j]:
                total += j*len(symbol_list)
                for s in symbol_list:
                    D_labeled_distribution[s] = j
            else:
                v = (profile[j]+1)/profile[j-1]*(j+1)
                total += v*len(symbol_list)
                for s in symbol_list:
                    D_labeled_distribution[s] = v
        else:
            total += j*len(symbol_list)
            for s in symbol_list:
                D_labeled_distribution[s] = j
    if Phi0>0:
        v = (profile[0]+1)/Phi0
        for s in range(k):
            if s not in D_labeled_distribution:
                D_labeled_distribution[s] = v
        total += profile[0]+1
    for s in D_labeled_distribution.keys():
        D_labeled_distribution[s] /= total
    
    return D_labeled_distribution
