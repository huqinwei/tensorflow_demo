import tensorflow as tf
from tensorflow.python import debug as tf_debug
sess = tf.Session()

sess = tf_debug.LocalCLIDebugWrapperSession(sess)
x = tf.placeholder(float,name = 'abcd')
def my_filter(tensor):
	return tensor.name()
sess.add_tensor_filter("my_filter",my_filter)
sess.run(x,{x:22.2})




