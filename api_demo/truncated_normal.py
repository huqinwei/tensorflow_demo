import tensorflow as tf
c = tf.truncated_normal(shape=[5,6])
c2 = tf.truncated_normal(shape=[52,62],mean=0,stddev=2)
c3 = tf.truncated_normal(shape=[5,6],mean=1,stddev=1)
c4 = tf.truncated_normal(shape=[5,6],mean=100,stddev=1)
d = tf.random_normal([5,6])#0,1
d2 = tf.random_normal([5,6],0,10)
d3 = tf.random_normal([5,6],100,1)
d4 = tf.random_normal([5,6],100,2)
d5 = tf.random_normal([23,26],0,2)
# d5_above4 = 1 if d5 > 4 else 0#TypeError: Using a `tf.Tensor` as a Python `bool` is not allowed. Use `if t is not None:` instead of `if t:` to test if a tensor is defined, and use TensorFlow ops such as tf.cond to execute subgraphs conditioned on the value of a tensor.
# d5_above4_sum = tf.reduce_sum(d5_above4)

with tf.Session() as sess:
    print('default:\n',sess.run(c))
    print('mean=100,stddev=2:\n',sess.run(c2))
    print('mean=1,stddev=1:\n',sess.run(c3))
    print('mean=2,stddev=1:\n',sess.run(c4))


    print('to compare,normal distribution')
    print('default:\n',sess.run(d))
    print('mean=0,stddev=10:\n',sess.run(d2))
    print('mean=100,stddev=1:\n',sess.run(d3))
    print('mean=100,stddev=2:\n',sess.run(d4))
    d5_ = sess.run(d5)
    d5_above4 = d5_ > 4
    counter = 0
    for boolean_list in d5_above4:
        for boolean in boolean_list:
            # print(boolean)
            if boolean == True:
                counter += 1
                # print('true')
    print('numbers > 2*stddev in d5:', counter)

    c2_ = sess.run(c2)
    c2_above4 = c2_ > 4
    counter = 0
    for boolean_list in c2_above4:
        for boolean in boolean_list:
            # print(boolean)
            if boolean == True:
                counter += 1
                # print('true')
    print('numbers > 2*stddev in c2:', counter)

    
