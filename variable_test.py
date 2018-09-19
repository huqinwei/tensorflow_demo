import tensorflow as tf
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0)
#print(node1,node2)

sess = tf.Session()
#print(tf.Session().run([node1,node2]))
#print(tf.Session().run(node1))
#print(tf.Session().run(node2))

#################################################

node3 = node1 + node2
#print("node3:",node3)
#print("run node3:",sess.run(node3))

#####################################3

W = tf.Variable([.3], dtype = tf.float32)
b = tf.Variable([-.3], dtype = tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W*x + b
print("W:",W)
#print(sess.run(linear_model,{x:[1,2,3,4]}))

init = tf.global_variables_initializer()
#print(sess.run(linear_model,{x:[1,2,3,4]}))

sess.run(init)
print("linear_model:",linear_model)
print(sess.run(linear_model,{x:[1,2,3,4]}))

######################################

W = tf.Variable(1)
assign_W = W.assign(10)
with tf.Session() as sess:
	sess.run(W.initializer)
	print(W.eval())
	sess.run(assign_W)
	print(W.eval())

#################################################
W = tf.Variable(10)
sess1 = tf.Session()
sess2 = tf.Session()
sess1.run(W.initializer)
sess2.run(W.initializer)
print("W add 1 = ",sess1.run(W.assign_add(1)))
print("W sun 2 = ",sess2.run(W.assign_sub(2)))
#print("W sun 2 = ",sess1.run(W.assign_sub(2)))#same session,same variable
sess1.close()
sess2.close()


