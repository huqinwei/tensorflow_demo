import tensorflow as tf

sess = tf.Session()

x = tf.constant([[1.,2.,3.],[4.,5.,6.]])
x = tf.reshape(x,[1,2,3,1])

x = tf.constant(0.1, shape = [1,2,3,1])
y = tf.Variable(x)
sess.run(tf.global_variables_initializer())
print(y)
print(sess.run(y))











