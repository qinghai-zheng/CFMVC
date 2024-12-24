import time
import scipy.io as scio
import numpy as np



count = 1
while(count < 1000):
    time.sleep(60)
    print('i')
    count += 1



# dataset='BBCSport'
# data = scio.loadmat('data/{}.mat'.format(dataset))
# print(data.keys())
# x1=scio.loadmat('data/{}.mat'.format(dataset))['X1']
# x2=scio.loadmat('data/{}.mat'.format(dataset))['X2']
# y=scio.loadmat('data/{}.mat'.format(dataset))['gt']
# x1=np.transpose(x1)
# x2=np.transpose(x2)
# y=np.squeeze(y)
# print(x1.shape,x1,sep='\n')
# print(x2.shape,x2,sep='\n')
# print(y.shape,y,sep='\n')








