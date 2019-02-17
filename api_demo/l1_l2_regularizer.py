import tensorflow as tf
CONST_SCALE = 0.5
w = tf.constant([[5.0, -2.0], [-3.0, 1.0]])
with tf.Session() as sess:
    print(sess.run(tf.abs(w)))
    print('preprocessing:', sess.run(tf.reduce_sum(tf.abs(w))))
    print('manual computation:', sess.run(tf.reduce_sum(tf.abs(w)) * CONST_SCALE))
    print('l1_regularizer:', sess.run(tf.contrib.layers.l1_regularizer(CONST_SCALE)(w))) #11 * CONST_SCALE

    print(sess.run(w**2))
    print(sess.run(tf.reduce_sum(w**2)))
    print('preprocessing:', sess.run(tf.reduce_sum(w**2) / 2))#default
    print('manual computation:', sess.run(tf.reduce_sum(w**2) / 2 * CONST_SCALE))
    print('l2_regularizer:', sess.run(tf.contrib.layers.l2_regularizer(CONST_SCALE)(w))) #19.5 * CONST_SCALE
