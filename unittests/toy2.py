import numpy as np
import scipy.stats as st 
import matplotlib.pyplot as plt 

x = st.norm.rvs(size = (1000,))
y = np.cos(x) + 0.2*st.norm.rvs()

plt.plot(x, y, 'x')
plt.show()

