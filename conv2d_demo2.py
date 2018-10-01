import tensorflow as tf
import numpy as np

k = np.float32([1, 4, 6, 4, 1])
k2 = np.inner(k,k)
print('inner:',k2)#inner equals to 1+16+36+16+1?yes!!!!!!!!!!!!!!
k = np.outer(k, k)
print('kernel:\n',k)
k5x5 = k[:,:] / k.sum() 
print('k5x5 shape:\n',k5x5.shape)
print('k5x5:\n',k5x5)




img = np.float32(np.random.randint(1,255,8*8))
img = img.reshape(8,8)
print(img)

#img = tf.nn.conv2d(img, k5x5, [1,1], 'SAME')
#print("after conv:\n",img)

#print("after conv:\n",tf.Session().run(img))













