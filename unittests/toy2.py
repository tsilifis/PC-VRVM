import numpy as np
import scipy.stats as st 
import matplotlib.pyplot as plt 

x = st.norm.rvs(size = (1000,))
y1 = np.sin(x) + 0.2*st.norm.rvs(size = (1000,))
y2 = np.cos(x) + 0.2*st.norm.rvs(size = (1000,))

plt.plot(x, y1, 'x')
plt.plot(x, y2, '+')
plt.show()

