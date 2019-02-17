#import add_layer
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
	layer_name = 'layer%s' % n_layer
	with tf.name_scope(layer_name):
		with tf.name_scope('weightss'):
			Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
			tf.summary.histogram(layer_name+'/weights', Weights)
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
			tf.summary.histogram(layer_name+'/biases', biases)
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs,Weights), biases)
			tf.summary.histogram(layer_name+'/Wx_plus_b', Wx_plus_b)
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
		tf.summary.histogram(layer_name+'/outputs', outputs)
		return outputs
def compute_accuracy(v_xs, v_ys):
	global prediction
	y_pre = sess.run(prediction,{xs:v_xs})	
	correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
	result = sess.run(accuracy,{xs:v_xs,ys:v_ys})
	return result


with tf.name_scope('inputs'):
	xs = tf.placeholder(tf.float32,[None,784],name='x_input')
	ys = tf.placeholder(tf.float32,[None,10],name='y_input')

prediction = add_layer(xs,784,1, n_layer = 1, activation_function = tf.nn.softmax)

with tf.name_scope('loss'):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
	tf.summary.scalar('cross_entropy', cross_entropy)
with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
init = tf.initialize_all_variables()

with tf.Session() as sess:
	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter("logs/",sess.graph)
	sess.run(init)
	for i in range(1000):
		batch_xs, batch_ys = mnist.train.next_batch(100)
		sess.run(train_step,{xs:batch_xs, ys:batch_ys})
		if i % 50 == 0:
			result = sess.run(merged, {xs:x_data,ys:y_data})
			writer.add_summary(result, i)
			print(compute_accuracy(mnist.test.images,mnist.test.labels))
