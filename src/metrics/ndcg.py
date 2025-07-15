import numpy as np

def ndcg_at_k(relevances, k):
    return np.sum(relevances[:k] / np.log2(np.arange(2, k+2)))