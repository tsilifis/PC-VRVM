import numpy as np
import scipy.stats as st 
import matplotlib.pyplot as plt 

x = st.norm.rvs(size = (1000,))
y = np.sin(x) + 0.1*st.norm.rvs()

plt.plot(x, y, '+')
plt.show()

