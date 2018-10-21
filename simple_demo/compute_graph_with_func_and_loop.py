import tensorflow as tf

#comput graph demo 1
x = tf.Variable(0)
print(id(x))
print(type(x))
# tf.assign_add(x,1, name = 'assign_add1')
x = x + 1
x = x.__add__(1)#TypeError: binary_op_wrapper() got an unexpected keyword argument 'name'

y = tf.Variable(0)
def func(x):
    for i in range(20):
        # tf.assign_add(x,1,name='run')
        x = x + 1
        # tf.assign_add(x, 1, name = 'assign_add_in_func')
        # assign_op = tf.assign(x, x + 1, name = 'assign_add_in_func')

    y = x
    return y
y = func(x)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    tf.summary.FileWriter('compute_graph_demo1', graph = sess.graph)
    # print(sess.run(func()))
    print(sess.run(x))
    # print(sess.run(assign_op))

    print(sess.run(y))



