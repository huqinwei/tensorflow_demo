import tensorflow as tf
t1 = tf.Variable(tf.ones([2,3,4], tf.int32))
t2 = tf.Variable(tf.ones([2,3,4], tf.int32))
print(t1)
print(t2)

t3 = tf.assign_add(t1, t2)
print(t3)

print(type(t1))
print(type(t1[1]))
t4 = t1[1] + t2[1]
t5 = t1[1][1][1] + t2[1][1][1]
# t6 = tf.scatter_add(t1[1][1][1] + t2[1][1][1])
# t42 = tf.assign_add(t1[1], t2[1])

# tf.Variable(tf.ones([2,3,4], tf.int32).assign_add(t1))#AttributeError: 'Tensor' object has no attribute 'assign_add'
# t3.assign_add(t1,t2)#AttributeError: 'Tensor' object has no attribute 'assign_add'
# tf.Tensor().assign_add()

sa_0_01 = tf.scatter_add(t1,[0],t2[0:1])#match
sa_1_01 = tf.scatter_add(t1,[1],t2[0:1])#not match
sa_1_12 = tf.scatter_add(t1,[1],t2[1:2])#match
# sa_0_12 = tf.scatter_add(t1,[0],t2[1:2])#not match
sa_0_12 = tf.scatter_add(t1,[0][0],t2[0][0][0])#not match
# sa_0_1 = tf.scatter_add(t1,[0],t2[1])#ValueError: Shapes must be equal rank, but are 2 and 3 for 'ScatterAdd_4' (op: 'ScatterAdd') with input shapes: [2,3,4], [1], [3,4].

print(t2[1])
print(t2[1:2])
# sa_0_1 = tf.scatter_add(t1,[0],[t2[1]])

# sa_01_12 = tf.scatter_add(t1,[0,1],t2[1:2])
# sa_02 = tf.scatter_add(t1,[1],t2[0:2])



with tf.Session() as sess:
    tf.global_variables_initializer().run()
    print('t1:', t1.eval())
    print('t2:', t2.eval())
    print('t3:', t3.eval())
    print('t1:', t1.eval())
    print('t2:', t2.eval())
    print('t4:', t4.eval())
    print('t5:', t4.eval())

    # print('sa_0_01:', sa_0_01.eval())
    # print('sa_1_01:', sa_1_01.eval())
    # print('t1:', t1.eval())
    # print('sa_12:', sa_1_12.eval())
    # print('sa_0_12:', sa_0_12.eval())
    # print('t1:', t1.eval())
    # print('sa_0_1:', sa_0_1.eval())
    # print('sa_01_12:', sa_01_12.eval())
    # print('ALL:sa_02:', sa_02.eval())








