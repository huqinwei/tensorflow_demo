#test:in this graph,update c,will a update?no
#can't use sess.run(a),it will execute a = a + 1
#just use sess.run(c) many times to prove it.
import tensorflow as tf
with tf.name_scope(name='init'):
    a = tf.Variable(3,dtype = tf.float32,name='init_a')
    b = tf.constant(3,dtype = tf.float32,name='init_b')
    c = tf.Variable(0,dtype=tf.float32)
a = a + 1
c = a + b
#update = tf.assign(c,a+b,name = 'update_c')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for i in range(5):
        #sess.run(update)
        print(sess.run(c))
    #print(sess.run(a))
    #print(sess.run(b))
    writer = tf.summary.FileWriter(logdir='graph_dir/',graph = sess.graph)