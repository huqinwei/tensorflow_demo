import tensorflow as tf
c = tf.truncated_normal(shape=[25,6])
c2 = tf.truncated_normal(shape=[25,6],mean=100,stddev=2)
c3 = tf.truncated_normal(shape=[25,6],mean=1,stddev=1)
c4 = tf.truncated_normal(shape=[25,6],mean=2,stddev=1)
d = tf.random_normal([25,6])#0,1
d2 = tf.random_normal([25,6],0,10)
d3 = tf.random_normal([25,6],100,1)
d4 = tf.random_normal([25,6],100,2)

with tf.Session() as sess:
    print('default:\n',sess.run(c))
    print('mean=100,stddev=2:\n',sess.run(c2))
    print('mean=1,stddev=1:\n',sess.run(c3))
    print('mean=2,stddev=1:\n',sess.run(c4))


    print('to compare,normal distribution')
    print('default:\n',sess.run(d))
    print('mean=0,stddev=1:\n',sess.run(d2))
    print('mean=100,stddev=1:\n',sess.run(d3))
    print('mean=100,stddev=2:\n',sess.run(d4))
    
