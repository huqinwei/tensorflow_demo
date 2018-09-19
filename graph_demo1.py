import tensorflow as tf
sess = tf.Session()
g = tf.Graph()
with g.as_default():
	c = tf.constant(30.0)
	assert c.graph is g
	assert c.graph is tf.get_default_graph()
	print(tf.get_default_graph())
	print(g)
c = tf.constant(4.0)
#print(sess.run(c))
assert c.graph is tf.get_default_graph()
#assert c.graph is g 
print(tf.get_default_graph())
print(g)
