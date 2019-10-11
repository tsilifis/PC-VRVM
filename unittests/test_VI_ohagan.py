import numpy as np 
import sys
import pandas as pd
import seaborn as sns
sns.set()
import chaos_basispy as cb
from toy import toy


# ---- Problem dimensionality -----
N = 1000
d = 10
order = 3
tr = 'TD'
q = 0.3
# ---- Setting the example: Coeffs, data and chaos model ------

sys.path.append('..')

from PC_VRVM import *


data={'xi':np.random.normal(size=(N,d))}
data['y'] = toy(data['xi'],d)


# --- RUN RELEVANCE VECTOR MACHINE FIT
chaos = ChaosModel(d, order)
a_0 = .2
b_0 = 1.
params = {'omega' : [1e-6, 1e-6], 'tau': [1e-6, 1e-6], 'pi': [ a_0, b_0 ]}#1./y.shape[0], (y.shape[0]-1.)/y.shape[0]]}

V0 = SparseVariationalOptimizer(chaos, data, params)


c_sol, omega_sol, tau_sol, z_sol, pi_sol, iters, elbo  = V0.optimize(tol = .0001)


df_c = pd.DataFrame(c_sol, columns=['mu','rho'])
df_o = pd.DataFrame(omega_sol, columns=['kappa','lambda'])
df_t = pd.DataFrame(tau_sol, columns=['m','n'])
df_z = pd.DataFrame(z_sol, columns=['pi'])
df_p = pd.DataFrame(pi_sol, columns=['alpha','beta'])
df_elbo = pd.DataFrame(elbo, columns = ['elbo'])

import matplotlib.pyplot as plt 
fig = plt.figure(figsize = (10, 5))
ax1 = fig.add_subplot(121)
ax1.plot(c_sol[:,0], 'x')
ax1.set_xscale('log')
ax2 = fig.add_subplot(122)
ax2.plot(z_sol, 'o')
ax2.set_xscale('log')
plt.show()

