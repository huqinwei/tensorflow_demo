import tensorflow as tf

state = tf.Variable(100,name='counter')
print(state)
print(state.name)

one = tf.constant(1)

new_value = tf.add(state,one)
update = tf.assign(state,new_value)

init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	print(sess.run(state))
	for _ in range(3):
		sess.run(update)
		print(sess.run(state))	
