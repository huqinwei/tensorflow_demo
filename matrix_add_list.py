#tensorflow add operation demo
#to test W*x + bias in batch condition

import tensorflow as tf

#batch_size = 2
Wx = tf.constant([[100,200,300],[400,500,600]])
print(Wx.shape)
b = tf.constant([1,1,1])
print(b.shape)

#every vector in batch plus b
Wx_plus_b = Wx + b

with tf.Session() as sess:
    print(sess.run(Wx))
    print(sess.run(Wx_plus_b))