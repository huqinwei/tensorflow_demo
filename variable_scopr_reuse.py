import tensorflow as tf

#demo1:conflict
# with tf.variable_scope(name_or_scope='name_scope_1'):
#     v1 = tf.get_variable(name='var',shape=[1])
#     v2 = tf.get_variable(name='var',shape=[1])
# print(v1.name)
# print(v2.name)

#demo2:rename to var_1
with tf.variable_scope(name_or_scope='var_scope_2'):
    v1 = tf.Variable(1,name='var')
    v2 = tf.Variable(1,name='var')
print(v1.name)
print(v2.name)

#demo3:rother scope
with tf.variable_scope(name_or_scope='var_scope_3'):
    v1 = tf.get_variable(name='var',shape=[1])
    with tf.variable_scope('var_scope_3'):
        v2 = tf.get_variable(name='var',shape=[1])
print(v1.name)
print(v2.name)


#demo4:name_scope
with tf.variable_scope('var_sco_4'):
    with tf.name_scope('name_scope'):
        v1 = tf.Variable([1],name='v1')
        v2 = tf.get_variable('v1',shape=[1],dtype=tf.int32)#v2 not exist
        v3 = v1 + v2
print(v1.name)
print(v2.name)
print(v3.name)
tf.Session().run(tf.global_variables_initializer())
# print(tf.Session().run(v2))
#print(tf.Session().run(v3))




#demo5:real reuse
with tf.variable_scope('var_s_5'):
    v1 = tf.get_variable(name='v1',shape=[1],dtype=tf.int32)
    tf.get_variable_scope().reuse_variables()
    v2 = tf.get_variable(name='v1',shape=[1],dtype=tf.int32)
print(v1.name)
print(v2.name)

#demo5.2:real reuse
with tf.variable_scope('var_s_6'):
    v1 = tf.get_variable(name='v1',shape=[1],dtype=tf.int32)
with tf.variable_scope('var_s_6', reuse=True):
    v2 = tf.get_variable(name='v1',shape=[1],dtype=tf.int32)
print(v1.name)
print(v2.name)

#demo5.3:real reuse
with tf.variable_scope('var_s_7',reuse=tf.AUTO_REUSE):
    v1 = tf.get_variable(name='v1',shape=[1],dtype=tf.int32)
    v2 = tf.get_variable(name='v1',shape=[1],dtype=tf.int32)
print(v1.name)
print(v2.name)







with tf.name_scope("increment"):
    zero64 = tf.constant(0, dtype=tf.int64)
    current = tf.Variable(zero64, name="incr", trainable=False, collections=[ops.GraphKeys.LOCAL_VARIABLES])


