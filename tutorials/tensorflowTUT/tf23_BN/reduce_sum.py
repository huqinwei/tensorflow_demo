import tensorflow as tf

x = tf.constant([[1,1,1],[1,1,1]])
tf.reduce_sum(x)
tf.reduce_sum(x,0)
