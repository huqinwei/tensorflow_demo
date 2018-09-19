import tensorflow as tf
sess = tf.Session()

INIT_PARAMS = [[5, 4],
               [5, 1],
               [2, 4.5]][2]
print(INIT_PARAMS)
print([INIT_PARAMS])
print(*INIT_PARAMS)
#a, b = [tf.Variable(initial_value=p, dtype=tf.float32) for p in INIT_PARAMS]
#a, b = tf.Variable(initial_value=INIT_PARAMS, dtype=tf.float32)
#a, b = tf.Variable(initial_value=[2,4.5], dtype=tf.float32)
a = tf.Variable(initial_value=INIT_PARAMS[0], dtype=tf.float32)
b = tf.Variable(initial_value=INIT_PARAMS[1], dtype=tf.float32)

init = tf.global_variables_initializer()
sess.run(init)
print(sess.run(a))
print(sess.run(b))
