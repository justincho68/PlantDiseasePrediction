from matplotlib import pyplot as plt
import numpy as np

mylist = [[1,0,1,0,1,0],[0,1,0,1,0,1]]

my_array = np.array(mylist)

plt.figure(figsize=(7,7))

plt.imshow(my_array, cmap='gray')

plt.show()