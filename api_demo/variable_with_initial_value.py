import tensorflow as tf
sess = tf.Session()

INIT_PARAMS_FATHER = [[5, 4],
               [5, 1],
               [2, 4.5]]

INIT_PARAMS = [[5, 4],
               [5, 1],
               [2, 4.5]][2]
print(INIT_PARAMS)
print([INIT_PARAMS])
print(*INIT_PARAMS)
print(INIT_PARAMS[0])
print(INIT_PARAMS[1])
#a, b = [tf.Variable(initial_value=p, dtype=tf.float32) for p in INIT_PARAMS]
#a, b = tf.Variable(initial_value=INIT_PARAMS, dtype=tf.float32)
#a, b = tf.Variable(initial_value=[2,4.5], dtype=tf.float32)
a = tf.Variable(initial_value=INIT_PARAMS[0], dtype=tf.float32)
b = tf.Variable(initial_value=INIT_PARAMS[1], dtype=tf.float32)
c = tf.Variable(initial_value=INIT_PARAMS, dtype=tf.float32)
d = tf.Variable(initial_value=INIT_PARAMS_FATHER, dtype=tf.float32)
print(d)

init = tf.global_variables_initializer()
sess.run(init)
print('a:\n',sess.run(a))
print('b:\n',sess.run(b))
print('c:\n',sess.run(c))
print('c:\n',sess.run(d))
