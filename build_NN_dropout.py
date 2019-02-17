#import add_layer
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

def add_layer(inputs, in_size, out_size, n_layer, activation_function=None):
	layer_name = 'layer%s' % n_layer
	with tf.name_scope(layer_name):
		with tf.name_scope('weights'):
			Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
#			tf.summary.histogram(layer_name+'/weights', Weights)
		with tf.name_scope('biases'):
			biases = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='b')
#			tf.summary.histogram(layer_name+'/biases', biases)
		with tf.name_scope('Wx_plus_b'):
			Wx_plus_b = tf.add(tf.matmul(inputs,Weights), biases)
			Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
#			tf.summary.histogram(layer_name+'/Wx_plus_b', Wx_plus_b)
		if activation_function is None:
			outputs = Wx_plus_b
		else:
			outputs = activation_function(Wx_plus_b)
#		tf.summary.histogram(layer_name+'/outputs', outputs)
		return outputs

#load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = .3)


with tf.name_scope('inputs'):
	keep_prob = tf.placeholder(tf.float32)
	xs = tf.placeholder(tf.float32,[None,64],name='x_input')
	ys = tf.placeholder(tf.float32,[None,10],name='y_input')

l1 = add_layer(xs, 64, 100, n_layer = 1, activation_function = tf.nn.tanh)
prediction = add_layer(l1,100,10, n_layer = 2, activation_function = tf.nn.softmax)

with tf.name_scope('loss'):
	cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
	tf.summary.scalar('loss(cross)', cross_entropy)

with tf.name_scope('train'):
	train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)
init = tf.initialize_all_variables()

with tf.Session() as sess:
	merged = tf.summary.merge_all()
	train_writer = tf.summary.FileWriter("logs/train",sess.graph)
	test_writer = tf.summary.FileWriter("logs/test",sess.graph)
	sess.run(init)
	for i in range(1000):
		sess.run(train_step,{xs:X_train, ys:y_train, keep_prob:0.4})
		if i % 50 == 0:
			train_result = sess.run(merged, {xs:X_train,ys:y_train, keep_prob:1})
			test_result = sess.run(merged, {xs:X_test,ys:y_test, keep_prob:1})
			train_writer.add_summary(train_result, i)
			test_writer.add_summary(test_result, i)
