import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x_data = np.linspace(-7,10,2500)
print(x_data[:10])
x_data2 = x_data[:,np.newaxis]
print('expand_dimension',x_data2[:10,:])
#x_data3 = x_data3[np.newaxis,:]
#print(x_data3[:,:20])
np.random.shuffle(x_data2)
##print('after shuffle:\n',x_data2[:10,:])
noise = np.random.normal(0,8,x_data2.shape)
#print(noise)
##print(np.mean(x_data2))
##print(np.mean(noise))

#y=x^2 - 5
y_data = np.square(x_data2) - 5 + noise
#print(y_data[:10,:])
#plt.scatter(x_data2,y_data)
#plt.show()


###this is tf23's x_data
xs2 = x_data2
layers_inputs2 = [xs2]
print("before\n",layers_inputs2)
layers_inputs2.append(x_data)
print('after append',layers_inputs2)

print(len(layers_inputs2))
print('[0]:\n',layers_inputs2[0])
#print('[1]:\n',layers_inputs2[1])

##xs = tf.placeholder(tf.float32,[None,1])
##layers_inputs = [xs]
##with tf.Session() as sess:
##    layers_inputs_ = sess.run(layers_inputs,{xs:x_data2})
##    print(layers_inputs_)
##    print(len(layers_inputs_))
