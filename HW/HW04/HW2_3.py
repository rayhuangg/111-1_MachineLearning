#%%
import scipy.io
import numpy as np
import matplotlib.pyplot as plt


mat = scipy.io.loadmat('04HW2_noisy.mat')
x = np.array(mat['X'])
face = x[:,10].reshape(28, 20)

plt.imshow(face, cmap='gray')
plt.show()


