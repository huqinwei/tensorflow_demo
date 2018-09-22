import numpy as np
import tensorflow as tf

value = [0,1,2,3,4,5,6,7]
init = tf.constant_initializer(value)
'''
print('fitting shape:')
with tf.Session():
    x = tf.get_variable('x',shape=[2,4],initializer=init)
    x.initializer.run()
    #sess.run(x.initializer)
    print(x.eval())

print('larger shape:')
with tf.Session():
    x = tf.get_variable('x',shape=[3,4],initializer=init)
    x.initializer.run()
    print(x.eval())
print('smaller shape:')
with tf.Session():
    x = tf.get_variable('x', shape=[2, 3], initializer=init)#ValueError
    x.initializer.run()
    print(x.eval())

'''

print('shape vertification:')
init_verify = tf.constant_initializer(value,verify_shape=True)
with tf.Session():
    x = tf.get_variable('x', shape=[3, 4], initializer=init_verify)#shape not fit
    x.initializer.run()
    print(x.eval())