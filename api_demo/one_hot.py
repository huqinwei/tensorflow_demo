#python
import tensorflow as tf

sess = tf.Session()

#demo1
indices = [0, 1, 2]#indices is so-called y-labels
depth_below = 2
depth_equal = 3
depth_above = 4
tf_oh_below = tf.one_hot(indices, depth_below)  # output: [3 x 2]
tf_oh_equal = tf.one_hot(indices, depth_equal)  # output: [3 x 3]
tf_oh_above = tf.one_hot(indices, depth_above)  # output: [3 x 4]
print(tf_oh_below)
print(sess.run(tf_oh_below))#bad
print(tf_oh_equal)
print(sess.run(tf_oh_equal))
print(tf_oh_above)
print(sess.run(tf_oh_above))#useless but work

#demo2
##indices is so-called y-labels,defaut start with 0 ,non-negative value
indices = [0, 2, -1, 1]#indice -1 is useless
depth = 3
depth2 = 6
tf_oh_on_off = tf.one_hot(indices, depth,
         on_value=5.0, off_value=0.0,
         axis=-1)  # output: [4 x 3]
tf_oh_on_off2 = tf.one_hot(indices, depth2,
         on_value=5.0, off_value=0.0,
         axis=-1)
tf_oh_on_off3 = tf.one_hot(indices, depth2,
         on_value=5.0, off_value=-1.0,
         axis=-1)
tf_oh_on_off4 = tf.one_hot(indices, depth2,
         on_value=666.0, off_value=0.0,
         axis=-1)
print(tf_oh_on_off)
print(sess.run(tf_oh_on_off))
print(tf_oh_on_off2)
print(sess.run(tf_oh_on_off2))
print(tf_oh_on_off3)
print(sess.run(tf_oh_on_off3))
print(tf_oh_on_off4)
print(sess.run(tf_oh_on_off4))

#demo3 shape changed,complex,todo~~~~~~~~~~~~~~~~~~~~~~~~~~~
indices = [[0, 2], [1, -1]]
depth = 3
depth2 = 6
tf_oh_axis_neg_one = tf.one_hot(indices, depth,
         on_value=1.0, off_value=0.0,
         axis=-1)  # output: [2 x 2 x 3]
tf_oh_axis_zero = tf.one_hot(indices, depth,
         on_value=1.0, off_value=0.0,
         axis=0)  # output: [2 x 2 x 3]
tf_oh_axis_one = tf.one_hot(indices, depth,
         on_value=1.0, off_value=0.0,
         axis=1)  # output: [2 x 2 x 3]

tf_oh_axis_one2 = tf.one_hot(indices, depth2,
         on_value=1.0, off_value=0.0,
         axis=1)  # output: [2 x 2 x 3]

print(tf_oh_axis_neg_one)
print(sess.run(tf_oh_axis_neg_one))
print(tf_oh_axis_zero)
print(sess.run(tf_oh_axis_zero))
print(tf_oh_axis_one)
print(sess.run(tf_oh_axis_one))
print(tf_oh_axis_one2)
print(sess.run(tf_oh_axis_one2))
