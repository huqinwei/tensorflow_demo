import tensorflow as tf
import numpy as np
#with tf.variable_scope("variables_1"): 
#	W = tf.Variable([[1,2,3],[3,4,5]], dtype = tf.float32, name='weights')
#	b = tf.Variable([[1,2,3]],dtype = tf.float32,name='biases')
#with tf.variable_scope("variables_2"):
#	W = tf.Variable([[1,2,3],[3,4,5]], dtype = tf.float32, name='weights')
#	b = tf.Variable([[1,2,3]],dtype = tf.float32,name='biases')

init = tf.global_variables_initializer()

#saver = tf.train.Saver()

with tf.Session() as sess:
	sess.run(init)
#	save_path = saver.save(sess, "my_net/save_net.ckpt")
#	print('Save to path:',save_path)



#same shape and same type and same name?
#W2 = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32,name='weights2')#wrong name
#b2 = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32,name='biases2')
#W2 = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32,name='weights')
#b2 = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32,name='biases')
#b3 = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32,name='biases3')#wrong name too!WTF!
#with tf.variable_scope("variables_1"):
with tf.variable_scope("variables_1"):
	W2 = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32,name='weights')
	b2 = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32,name='biases')
with tf.variable_scope("variables_2"):
	W22 = tf.Variable(np.arange(6).reshape((2,3)), dtype=tf.float32,name='weights')
	b22 = tf.Variable(np.arange(3).reshape((1,3)), dtype=tf.float32,name='biases')

init2 = tf.global_variables_initializer()

#no need init
loader = tf.train.Saver()
load_path = "my_net/save_net.ckpt"
with tf.Session() as sess:
#3 way to initialize
#	sess.run(init)#error:not work,sequence problem
	#sess.run(init2)
#	sess.run(init2)
	loader.restore(sess, load_path)

	print("weights2:",sess.run(W2))
	print("weights2:",sess.run(W22))
#	print("biases2:",sess.run(b2))

