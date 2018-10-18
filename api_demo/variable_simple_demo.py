import tensorflow as tf
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
node3 = node1 + node2
print("Tensor node3:",node3)

sess = tf.Session()
print('node1:', tf.Session().run(node1))
print('node2:', tf.Session().run(node2))
print('[node1,node2]:', tf.Session().run([node1,node2]))
print("node3:",sess.run(node3))

#######################################
#multiply and add
W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b

init = tf.global_variables_initializer()
sess.run(init)
print('linear_model:',sess.run(linear_model,{x:[1,2,3,4]}))

######################################
#assign
W2 = tf.Variable(1)
assign_W2 = W2.assign(10)
with tf.Session() as sess:
	sess.run(W2.initializer)
	print('W2:', W2.eval())
	sess.run(assign_W2)
	print('W2:', W2.eval())

#################################################
#same variable in different session
W3 = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W3.initializer)
sess2.run(W3.initializer)
print("session1:W add 1 = ",sess1.run(W3.assign_add(1)))
print("session1:W sun 2 = ",sess2.run(W3.assign_sub(2)))
print("session1:W add 1 = ",sess1.run(W3.assign_add(1)))
print("session1:W sun 2 = ",sess2.run(W3.assign_sub(2)))
#print("W sun 2 = ",sess1.run(W.assign_sub(2)))#same session,same variable
sess1.close()
sess2.close()


