#to test tf.add_n and error
#[<tf.Tensor 'conv1/weight_loss:0' shape=(5, 5, 3, 64) dtype=float32>, <tf.Tensor 'conv2/weight_loss:0' shape=(5, 5, 64, 64) dtype=float32>, <tf.Tensor 'fc1/weight_loss:0' shape=(2304, 384) dtype=float32>, <tf.Tensor 'fc2/weight_loss:0' shape=(384, 192) dtype=float32>, <tf.Tensor 'softmax_linear/weight_loss:0' shape=(192, 10) dtype=float32>, <tf.Tensor 'cross_entropy:0' shape=() dtype=float32>]
#tensorflow.python.framework.errors_impl.InvalidArgumentError: Shapes must be equal rank, but are 2 and 0
#	From merging shape 4 with other shapes. for 'total_loss' (op: 'AddN') with input shapes: [5,5,3,64], [5,5,64,64], [2304,384], [384,192], [192,10], [].


import tensorflow as tf
import numpy as np

#simple demo 1
input1 = tf.constant([1.0,2.0,3.0])
input2 = tf.Variable(tf.random_uniform([3]))
output = tf.add_n([input1, input2])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(input1))
    print(sess.run(input2))
    print(sess.run(input1 + input2))#new op defined!!!!
    print(sess.run(output))

#demo1.2:different shape
# ValueError: Dimension 0 in both shapes must be equal, but are 4 and 3. Shapes are [4] and [3].
# 	From merging shape 0 with other shapes. for 'AddN' (op: 'AddN') with input shapes: [4], [3].
# input1 = tf.constant([1.0,2.0,3.0,5.0])
# input2 = tf.Variable(tf.random_uniform([3]))
# output = tf.add_n([input1, input2])
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(input1))
#     print(sess.run(input2))
#     print(sess.run(input1 + input2))#new op defined!!!!
#     print(sess.run(output))

#demo1.3:different rank--this is what my program occured
# ValueError: Shapes must be equal rank, but are 2 and 1
# 	From merging shape 0 with other shapes. for 'AddN' (op: 'AddN') with input shapes: [2,3], [3].
# input1 = tf.constant([[1.0,2.0,3.0],[3.,4.,5.]])
# input2 = tf.Variable(tf.random_uniform([3]))
# output = tf.add_n([input1, input2])
#
# with tf.Session() as sess:
#     sess.run(tf.initialize_all_variables())
#     print(sess.run(input1))
#     print(sess.run(input2))
#     print(sess.run(input1 + input2))#new op defined!!!!
#     print(sess.run(output))

#demo2 with collection--like a reduce_sum
# tf.add_to_collection('losses', tf.constant(2.2))
# tf.add_to_collection('losses', tf.constant(3.))
#
# #demo2.2
# #does collection equals to scope?
# with tf.name_scope(name = 'losses') as scope:
#     v1 = tf.Variable([1.0],name = 'variable1')
#
# with tf.Session() as sess:
#     print(sess.run(tf.get_collection('losses')))
#     print(sess.run(tf.add_n(tf.get_collection('losses'))))
#     #demo 2.2
#     sess.run(tf.global_variables_initializer())
#     print(v1)
#     print(sess.run(v1))

















