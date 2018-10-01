#strided slice
import tensorflow as tf
sess = tf.Session()
#easy demo
t = tf.constant([1,2,3,4,5,6,7,8])
t1 = tf.strided_slice(t,[1],[-1])
t2 = tf.strided_slice(t,[3],[999])#stands for -0
t3 = tf.strided_slice(t,[1],[3])
t4 = tf.strided_slice(t,[0],[-1])
#t41 = tf.strided_slice(t,[0],[-1],[0])#must be non-zero
t42 = tf.strided_slice(t,[0],[-1],[1])
t43 = tf.strided_slice(t,[0],[-1],[2])
t44 = tf.strided_slice(t,[0],[-1],[3])

t5 = tf.strided_slice(t,[3],[5])
t52 = tf.strided_slice(t,[5],[3])
t53 = tf.strided_slice(t,[5],[3],[-1])
print(sess.run(t1))
print(sess.run(t2))
print(sess.run(t3))
print(sess.run(t4))
#print(sess.run(t41))
print(sess.run(t42))
print(sess.run(t43))
print(sess.run(t44))
print(sess.run(t5))
print(sess.run(t52))
print(sess.run(t53))

# #demo 2
# t = tf.constant([[[11, 12, 13], [21, 22, 23]],
#                  [[31, 32, 33], [41, 42, 43]],
#                  [[51, 52, 53], [61, 62, 63]]])
# print(t)
# t1 = tf.strided_slice(input_ = t,begin = [1,0,0], end = [2,1,3], strides = [1,1,1])
# print(sess.run(t1))
# t2 = tf.strided_slice(t, [1, -1, 0], [2, -3, 3], [1, -1, 1])
# print(sess.run(t2))
# t22 = tf.strided_slice(t, [1, -1, 0], [2, -3, 2], [1, -1, 1])
# print(sess.run(t22))
# t23 = tf.strided_slice(t, [1, -1, 0], [1, -3, 2], [1, -1, 1])
# print(sess.run(t23))
# t24 = tf.strided_slice(t, [1, -1, 0], [2, -3, 3], [1, -1, 2])
# print(sess.run(t24))
