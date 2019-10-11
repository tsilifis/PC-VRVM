"""
Author: Panagiotis Tsilifis
Date: 11.10.2019

Modified O'Hagan function [1]:
f(x) = a1^T * x + a2^T * sin(x) + a3^T * cos(x) + cos(x)^T * M * sin(x)

[1] Oakley, J.E. & O'Hagan, A. (2004). Probabilistic sensitivity analysis of complex 
    models: a Bayesian approach. Journal of the Royal Statistical Society: Series B 
    (Statistical Methodology), 66(3), 751-769.
"""

def toy(x, d=5):
    import numpy as np
    import scipy.stats as st 
    np.random.seed(123456)

    if x.ndim==1: x= x.reshape(1,-1) 
    assert x.shape[1]==d
    out = np.empty(x.shape[0])
    a1 = np.hstack([st.uniform.rvs(scale = 1, size = (d-3,)), st.uniform.rvs(loc=1.5, scale = 2, size = (3,)) ])
    a2 = np.hstack([st.uniform.rvs(scale = 1, size = (d-3,)), st.uniform.rvs(loc=1.5, scale = 2, size = (3,)) ])
    a3 = np.hstack([st.uniform.rvs(scale = 1, size = (d-3,)), st.uniform.rvs(loc=1.5, scale = 2, size = (3,)) ])

    M = st.uniform.rvs(scale = 2, size = (d,d))
    
    # the first three terms: a1, a2 and a3
    out = np.dot(x,a1) + np.dot(np.sin(x),a2) + np.dot(np.cos(x),a3)
    
    # the quadratic term for each realization
    for realization in range(len(out)):
        out[realization] += np.dot(np.cos(x[realization,]).T, np.matmul(M, np.sin(x[realization,]) ))  

    print out.mean(), out.std()
    return out