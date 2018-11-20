import numpy as np

def cross_entropy(Y, P):
	"""A function that takes as input two lists Y, P,
	and returns the float corresponding to their cross-entropy.

	"""
    Y = np.float_(Y)
    P = np.float_(P)
    entropy = np.sum((Y * np.log(P)) + ((1-Y)* np.log(1-P)))
    return -1*entropy