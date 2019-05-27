import numpy as np
log = np.log
exp = np.exp

def KL_divergence(D_dist_estimate,distribution):
    KL = 0
    for j in range(distribution.size):
        prob = distribution[j]
        KL += prob*log(prob/D_dist_estimate[j])
    return KL

def get_distribution(dist_name, k):
    k = int(k)
    if dist_name == 'uniform':
        raw_distribution = [1] * k
    elif dist_name == 'two-step':
        raw_distribution = [1] * (k//2) + [5] * (k-k//2)
    return raw_distribution

def generate_distribution(distribution_name,k):
    raw_distribution = get_distribution(distribution_name,k)
    sum_raw = sum(raw_distribution)
    distribution = sorted([float(y)/float(sum_raw) for y in raw_distribution])
    return np.array(distribution)

def get_samples(distribution, n):
    values = np.random.rand(n)
    values.sort()
    cumulative = np.cumsum(distribution)
    z = 0
    k = len(distribution)
    freq = [0] * k
    for x in values:
        while x > cumulative[z]:
            z+=1
        freq[z]+=1
    freq_positive = []
    D_freq_positive = {}
    for j in range(k):
        x = freq[j]
        if x>0:
            freq_positive.append(x)
            if x in D_freq_positive:
                D_freq_positive[x].append(j)
            else:
                D_freq_positive[x] = [j]
    return [freq_positive, D_freq_positive]

def generate_sample(distribution,n):
    [multiplicity, D_multiplicity] = get_samples(distribution,n)
    return [np.array(multiplicity), D_multiplicity]
