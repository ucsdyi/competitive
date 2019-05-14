import numpy as np
import ctypes
import scipy.stats
import matplotlib.pyplot as plt
from collections import Counter
from scipy.interpolate import interp1d
from Others import KL_divergence, generate_distribution, generate_sample
from New_Estimator import D_new
log = np.log
exp = np.exp

##### Code for the paper "Doubly-Competitive Distribution Estimation" by Yi Hao and Alon Orlitsky
##### Paper to appear at ICML 2019.
##### Plots will be saved to local dirve.

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



##### Experiments
def experiment(distribution_name, r):
    k = 10000 # Alphabet size
    nlist = range(100000,1000001,100000) # Number of samples
    New, GT = [], []
    distribution = generate_distribution(distribution_name,k)

    for n in nlist:
        print("Number of Samples for the experiment: "+str(n))
        New_est, GT_est = 0, 0
        for j in range(r):
            [multiplicity, D_multiplicity] = generate_sample(distribution,n)
            profile_counter = Counter(multiplicity)
            profile = [profile_counter[i] for i in range(1,max(multiplicity)+1)]
            
            # Improved Good-Turing estimator in (Orlitsky & Suresh, 2015).
            labeled_GT = Good_Turing(D_multiplicity,profile,k)
            # The proposed estimator.
            labeled_New = D_new(D_multiplicity,profile,k)
            
            GT_est += KL_divergence(labeled_GT,distribution)
            New_est += KL_divergence(labeled_New,distribution)

        GT.append(GT_est/r)
        New.append(New_est/r)
       
    print("\n Done.\n")
    
    fig = plt.figure()
    
    xnew = np.linspace(min(nlist),max(nlist),200,endpoint=True)
    New_int = interp1d(nlist,New,kind='cubic')
    plt.plot(xnew,New_int(xnew),'cyan',linewidth=3)
    GT_int = interp1d(nlist,GT,kind='cubic')
    plt.plot(xnew,GT_int(xnew),'darkviolet',linewidth=3)
    
    legend = ['proposed','Good-Turing+empirical']
    plt.legend(legend, loc='upper right')
    plt.xlabel('Number of samples', fontsize=10)
    plt.ylabel('Average KL divergence', fontsize=10)
    save_name = distribution_name+'.jpg'
    fig.savefig(save_name)

def main(): 
    # Options for distribution_name: 'uniform','two-steps'.
    D_list = ['uniform', 'two-steps']
    r = 1 # number of independent repetitions
    for distribution_name in D_list:
        print("Performing experiments for "+distribution_name+".\n")
        experiment(distribution_name,r)

if __name__=="__main__":
    main()
