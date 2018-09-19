import tensorflow as tf
g=tf.Graph()
with g.as_default():
	c = tf.constant(5.0)
	assert c.graph is g

with tf.Graph().as_default() as g:
	c = tf.constant(5.0)
	assert c.graph is g

