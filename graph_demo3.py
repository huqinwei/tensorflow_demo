import tensorflow as tf
def my_func(pred,tensor):
	t = tf.matmul(tensor,tensor)
	with tf.control_dependencies([pred]):
		return t
