# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)#tf.multiply is not matrix multiply
output2 = tf.matmul(input1,input2)#this is matrix multiply

with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [7.], input2: [2.]}))
    print(sess.run(output, feed_dict={input1: [7.,6.], input2: [2.]}))
    print(sess.run(output, feed_dict={input1: [7.,6.], input2: [2.,5.]}))

    #print(sess.run(output2, feed_dict={input1: [7.,6.], input2: [[2.],[5.]]}))#error:in[0]'s ndims is [2],not[1,2]
    print(sess.run(output2, feed_dict={input1: [[7.,6.]], input2: [[2.],[5.]]}))#now in[0]'s ndims is [1,2]
    print(sess.run(output2, feed_dict={input1: [[7.],[6.]], input2: [[2.,5.]]}))
