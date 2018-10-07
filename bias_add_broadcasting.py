#to test broadcasting in  bias_add

import tensorflow as tf
import numpy as np

a = tf.Variable(np.arange(12).reshape(3,4))
b = tf.constant([1,2,3,4],dtype=tf.int64)

bias_add1 = tf.nn.bias_add(a,b)
bias_add2 = a + b

#sess = tf.Session()
with tf.Session() as sess:
    tf.global_variables_initializer().run()

    print(sess.run(a))
    print(sess.run(bias_add1))
    print(sess.run(bias_add2))





