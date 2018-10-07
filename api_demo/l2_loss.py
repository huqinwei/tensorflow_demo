#weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
#l2_loss
#output = sum(t**2) / 2

import tensorflow as tf

a = tf.constant([1.])
b = tf.constant([1.2])
c = tf.constant([1.,2.,3.])
output = tf.nn.l2_loss(a)
output2 = tf.nn.l2_loss(b)
output3 = tf.nn.l2_loss(c)

with tf.Session() as sess:
    print(sess.run(output))
    print(sess.run(output2))
    print(sess.run(output3))









