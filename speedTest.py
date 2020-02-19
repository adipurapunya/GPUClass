import numpy as np;
import pycuda.autoinit
from pycuda import gpuarray

from time import time

host_data = np.float32(np.random.random(50000000))

t1 = time()
host_data_2x = host_data * np.float32(2)
t2 = time()

print('Total time to compute on CPU: ', (t2-t1))

device_data = gpuarray.to_gpu(host_data)


t3 = time()
device_data_2x = device_data * np.float32(2)
t4 = time()

from_device = device_data_2x.get()
print('Total time to compute on GPU: ', (t4-t3))

print ('is  the host computation the same as the GPU computation ? : {} '.format(np.allclose(from_device, host_data_2x)))