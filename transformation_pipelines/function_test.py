import numpy as np

a = np.array([[1, 2, 3], [4, 5, 6]])

new_a_x = np.concatenate((a, a), axis=0)

print(new_a_x)

r = np.r_[a, a]

print(r)