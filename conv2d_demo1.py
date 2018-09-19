import tensorflow as tf
import numpy as np

k = np.float32([1, 4, 6, 4, 1])
k = np.outer(k, k)
print('kernel:\n',k)
eye = np.eye(3, dtype=np.float32)
#print("eye:\n",eye)
#print("k.sum():",k.sum())
k5x5 = k[:,:,None,None] / k.sum() * np.eye(3, dtype=np.float32)
pre_eye_k5x5 = k[:,:,None,None] / k.sum() 
#print('k5x5_pre_eye shape:\n',pre_eye_k5x5.shape)
#print('k5x5_pre_eye:\n',pre_eye_k5x5)
print('k5x5 shape:\n',k5x5.shape)
print('k5x5:\n',k5x5)




img = np.float32(np.random.randint(1,255,8*8*3))
img = img.reshape(1,8,8,3)
#img = tf.expand_dims(img,0)#
print(img)

img = tf.nn.conv2d(img, k5x5, [1,2,2,1], 'SAME')
print("after conv:\n",img)

print("after conv:\n",tf.Session().run(img))













