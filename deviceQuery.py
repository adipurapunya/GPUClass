import pycuda.driver as drv

drv.init()

print("Total Device is %s" % (drv.Device.count()))