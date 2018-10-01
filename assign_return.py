import tensorflow as tf

v = tf.Variable(3,name='v')
v2 = v.assign(5)
v3 = tf.assign(v,6)#change v,and return new value to v3!!!!!!!
update = tf.assign(v,10)


sess = tf.InteractiveSession()
sess.run(v.initializer)
print(sess.run(v))

print(sess.run(v2))
print(sess.run(v3))
print(sess.run(v))
sess.run(update)
print(sess.run(v))

#@###########################################
#demo2

# v = tf.Variable(3,name='v')
# v.assign(5)
# v2 = v.assign(5)
#
# sess = tf.InteractiveSession()
# sess.run(v.initializer)
# print(sess.run(v))
#
# print(sess.run(v2))
#@###########################################
#demo3

# v = tf.Variable(3,name='v')
# # v = v.assign(5)#weird error:Tensor has no attribute 'assign'
# #v.assign(5)
# #tf.assign(v,5)
# v = 5#weird error:int has no attribute 'assign'
# #tf.assign() actually called Tensor obj's assign()??
# v2 = v.assign(5)
#
# sess = tf.InteractiveSession()
# sess.run(v.initializer)
# print(sess.run(v))
#
# print(sess.run(v2))
