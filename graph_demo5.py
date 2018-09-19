import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
#tf.ConfigProto(allow_soft_placement=False)
a = tf.constant([1.0,2.0],name = 'a')
b = tf.constant([1.0,2.0],name = 'b')
g = tf.Graph()
#with g.device('/gpu:0'):
#with g.device('/gpu:0'):
with g.device('/gpu:3'):
	result = a + b
	print(sess.run(result))
print(a.graph is tf.get_default_graph())

