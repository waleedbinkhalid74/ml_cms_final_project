import numpy as np
from scipy.sparse import dia_matrix
from scipy.sparse.linalg import inv


def dmd(data: np.array):
    
    X = data[:,:-1]
    Y = data[:,1:]
    
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    
    inv_s = np.linalg.pinv(np.diag(s))
    
    A_tilda = u.T @ Y @ vh.T @ inv_s
    
    w, v = np.linalg.eig(A_tilda)
    phi = u @ v
    
    return A_tilda, w, phi