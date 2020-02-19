from timeit import default_timer

import numpy as np;
import pycuda.autoinit
from pycuda import gpuarray

x_host = np.array([1,2,3], dtype=np.float32)
y_host = np.array([1,1,1], dtype=np.float32)
z_host = np.array([2,2,2], dtype=np.float32)

x_device = gpuarray.to_gpu(x_host)
y_device = gpuarray.to_gpu(y_host)
z_device = gpuarray.to_gpu(z_host)

#print(x_host + y_host)
#print((x_device + y_device).get())

#print(x_host ** z_host)
#print((x_device ** z_device).get())

#print(x_host / x_host)
#print((x_device / x_device).get())

#print(z_host - x_host)
#print((z_device - x_device).get())

#print(z_host / 2)
#print((z_device / 2).get())

#print(z_host - 1)
#print((z_device - 1).get())