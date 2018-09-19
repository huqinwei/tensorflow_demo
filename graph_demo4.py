import tensorflow as tf
g1 = tf.Graph()
with g1.as_default():
	v=tf.get_variable('v',initializer=tf.zeros_initializer()(shape=[2,2]))
#	v=tf.get_variable('v')
g2 = tf.Graph()
with g2.as_default():
	v=tf.get_variable('v',initializer=tf.ones_initializer()(shape=[2,2]))

with tf.Session(graph=g1) as sess:
	tf.global_variables_initializer().run()
	with tf.variable_scope('',reuse=True):
		print(sess.run(tf.get_variable('v')))
#		print(tf.GraphKeys.VARIABLES())


with tf.Session(graph=g2) as sess:
	tf.global_variables_initializer().run()
	with tf.variable_scope('',reuse=True):
		print(sess.run(tf.get_variable('v')))
