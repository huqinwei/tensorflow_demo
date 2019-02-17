import tensorflow as tf
import numpy as np


W = tf.Variable([1,3,4])
W2 = tf.Variable(tf.zeros([3,3]))
W3 = tf.Variable(np.arange(12).reshape(3,4))
W4 = tf.Variable(initial_value = 0,expected_shape=[3,4])
W5 = tf.Variable(initial_value = np.arange(12).reshape(3,4),expected_shape=[3,4])
#once you feed a initial_value to Variable,the expected_shape is nonsense!!
W52 = tf.Variable(initial_value = np.arange(12).reshape(4,3),expected_shape=[3,4])
input = tf.placeholder(dtype = tf.float32, shape = [3,4])
output = input
W6 = tf.Variable(initial_value = np.arange(12).reshape(3,4),expected_shape=[3,4])
#once you feed a value to Variable,the initial_value is nonsense!!
W7 = tf.Variable(initial_value = np.arange(3),expected_shape=[3,4])
#once you feed a value to Variable,the expected_shape is nonsense!!
W8 = tf.Variable(initial_value = np.arange(3),expected_shape=[2,6])
W6 = input
W7 = input
W8 = input

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(W))
    print(sess.run(W2))
    print(sess.run(W3))
    print(sess.run(W4))
    print(sess.run(W5))
    print('W52:\n',sess.run(W52))
    #print(sess.run(output, {input:100}))#ValueError
    print(sess.run(output, {input:np.arange(12).reshape(3,4)}))
    print(sess.run(W6, {input:np.arange(12).reshape(3,4)}))
    print(sess.run(W7, {input:np.arange(12).reshape(3,4)}))
    print(sess.run(W8, {input:np.arange(12).reshape(3,4)}))


