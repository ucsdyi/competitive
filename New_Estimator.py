import numpy as np
log = np.log
exp = np.exp

##### The new estimator
def Emu(mu1, mu, T):
    A = 1/T
    B = 1
    for j in range(1,mu1+1):
        B += log(j)
    for j in range(1,mu+1):
        B -= log(j)
    B += (mu-mu1)*log(mu1)
    A *= exp(B)
    return A

def D_new(D_multiplicity, profile, k):
    c1, c2, c3 = 2, 0.5, 1 # Hyper-parameters, have not been optimized
    n = 0
    Key = D_multiplicity.keys()
    
    for j in Key:
        n += j*len(D_multiplicity[j])
    
    D_labeled_distribution = {}
    
    total = 0
    IC = 0
    for j in Key:
        symbol_list = D_multiplicity[j]
        if j<=c3*log(n) and profile[j-1]>c2*(log(n))**2: # Good-Turing component
            T = (profile[j]+1)*(j+1)
            total += T
            v = T/len(symbol_list)
            for s in symbol_list:
                D_labeled_distribution[s] = v
    
        elif j>c3*log(n) and profile[j-1]>c2*(log(n))**2: # Improved component
            IC += 1
            
            K = c1*np.sqrt((j+1)/log(n))
            Ej_add_1 = sum([profile[mu1-1]*Emu(mu1, j+1, K) for mu1 in range(int(j+1-K),j+1)])
            K = c1*np.sqrt(j/log(n))
            Ej = sum([profile[mu1-1]*Emu(mu1, j, K) for mu1 in range(int(j-K),j)])
            T = profile[j-1]*(j+1)*Ej_add_1/Ej+1
            total += T
            v = T/len(symbol_list)
            for s in symbol_list:
                D_labeled_distribution[s] = v

        elif profile[j-1]<=c2*(log(n))**2: # Empirical component
            T = j*len(symbol_list)
            total += T
            for s in symbol_list:
                D_labeled_distribution[s] = j
    Phi0 = 0
    for j in Key:
        Phi0 += len(D_multiplicity[j])
    Phi0 = k - Phi0
    if Phi0>0:     # Good-Turing component for mu = 0
        v = (profile[0]+1)/Phi0
        for s in range(k):
            if s not in D_labeled_distribution:
                D_labeled_distribution[s] = v
        total += profile[0]+1
    for s in D_labeled_distribution.keys():
        D_labeled_distribution[s] /= total
    
    print('The improved component was executed '+str(IC)+' times.\n')
    return D_labeled_distribution
