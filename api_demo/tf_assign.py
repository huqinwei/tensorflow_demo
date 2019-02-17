import tensorflow as tf


x1 = tf.Variable(0,dtype=tf.float32)
x2 = tf.Variable(0,dtype=tf.float32)
x3 = tf.Variable(0,dtype=tf.float32)
x4 = tf.Variable(0,dtype=tf.float32)

update1 = tf.assign(x1,x1+1)
update2 = tf.assign_add(x2,1)
update3 = tf.assign_add(x3,2)
update4 = tf.assign_sub(x4,1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(4):
        sess.run([update1,update2,update3,update4])
        print('assign x+1:',sess.run(x1))
        print('assign_add 1:',sess.run(x2))
        print('assign_add 2:',sess.run(x3))
        print('assign_sub 1:',sess.run(x4))