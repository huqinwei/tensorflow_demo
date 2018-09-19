import tensorflow as tf
import numpy as np

A=[[1,3,4,5,6]]
B = [[1,3,4],[2,4,1]]
C = [[1,3,4],[2,4,1],[100,33,44]]
with tf.Session() as sess:
	print(sess.run(tf.argmax(A,1)))
	print(sess.run(tf.argmax(B,1)))
	print(sess.run(tf.argmax(C,1)))
#	print(sess.run(tf.argmax(B,2)))









