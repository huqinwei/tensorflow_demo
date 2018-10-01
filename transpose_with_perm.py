#transpose demo
import numpy as np
import tensorflow as tf

input = tf.Variable(np.arange(12).reshape([2,3,2]))

#output_origin = tf.transpose(input,[0,1,2])
output_2 = tf.transpose(input,[0,2,1])
output_3 = tf.transpose(input,[2,1,0])
output_4 = tf.transpose(input,[1,2,0])

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(sess.run(input))
#print(sess.run(output_origin))
print(sess.run(output_2))
print(sess.run(output_3))
print(sess.run(output_4))



