import tensorflow as tf

t_input = tf.Variable([[1,2],[3,4]])

t_expand1 = tf.expand_dims(t_input, 0)

t_expand2 = tf.expand_dims(t_input, axis = 2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(sess.run(t_input))
    print(sess.run(t_expand1))
    print(sess.run(t_expand2))

