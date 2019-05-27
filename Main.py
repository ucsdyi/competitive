import numpy as np
import scipy.stats
from scipy.interpolate import interp1d
from collections import Counter
import matplotlib.pyplot as plt

from Others import KL_divergence, generate_distribution, generate_sample
from New_Estimator import D_new
from Good_Turing_Estimator import Good_Turing

log = np.log
exp = np.exp

##### Code for paper "Doubly-Competitive Distribution Estimation" by Hao & Orlitsky (ICML 2019)
##### Plots will be saved to the current working directory

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
            
            # Improved Good-Turing estimator in (Orlitsky & Suresh, 2015):
            labeled_GT = Good_Turing(D_multiplicity,profile,k)
            # Proposed estimator:
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
    
    legend = ['Proposed','Good-Turing+empirical']
    plt.legend(legend, loc='upper right')
    plt.xlabel('Number of samples', fontsize=10)
    plt.ylabel('Average KL divergence', fontsize=10)
    save_name = distribution_name+'.jpg'
    fig.savefig(save_name)

def main(): 
    # Options for distribution_name: 'uniform','two-step'
    D_list = ['uniform', 'two-step'] #List of distributions
    r = 1 # Number of independent trials
    for distribution_name in D_list:
        print("Performing experiments for "+distribution_name+".\n")
        experiment(distribution_name,r)

if __name__=="__main__":
    main()
