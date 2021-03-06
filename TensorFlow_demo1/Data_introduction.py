import pickle as p
import numpy as np
import os
import matplotlib.pyplot as ply
import pylab
CIFAR_DIR = "D:/user/Download/cifar-10-batches-py"
print(os.listdir(CIFAR_DIR))
with open(os.path.join(CIFAR_DIR,"data_batch_1"), 'rb') as f:
    data = p.load(f,encoding='iso-8859-1')
    print(type(data))
    print(data.keys())
    print(type(data['data']))
    print(type(data['labels']))
    print(type(data['batch_label']))
    print(type(data['filenames']))
    print(data['data'].shape)
    print(data['data'][0:2])
    print(data['labels'][0:2])
    print(data['batch_label'])
    print(data['filenames'][0:2])
    image_arr = data['data'][100]
    image_arr = image_arr.reshape((3,32,32))#32 32 3
    image_arr = image_arr.transpose((2,1,0))
    ply.imshow(image_arr)
    pylab.show()
