import tensorflow as tf

input = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)

#output = tf.mul(input,input2)
output = tf.multiply(input,input2)

with tf.Session() as sess:
	print(sess.run(output,{input:[3.],input2:[2.]}))
