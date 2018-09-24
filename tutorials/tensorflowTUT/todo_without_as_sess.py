# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf

matrix1 = tf.constant([[3, 3]])
matrix2 = tf.constant([[2],
                       [2]])
product = tf.matmul(matrix1, matrix2)  # matrix multiply np.dot(m1, m2)
'''
# method 1
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

init = tf.global_variables_initializer()
# method 2
with tf.Session() as sess:
    result2 = sess.run(product)
    #result2 = product.run()#tensor has no attribute run
    init.run()#this already have a run function inside,initializer() is an op,not tensor
    print(result2)

'''


# method 3?
with tf.Session():#if i don't use as sess,how to run sess.run()?????
    result2 = run(product)
    #result2 = product.run()#tensor has no attribute run
    init.run()#this already have a run function inside,initializer() is an op,not tensor
    print(result2)

###################################################################
#with as demo

# !/usr/bin/env python
# with_example01.py
'''
class Sample:
    def __enter__(self):
        print("In __enter__()")
        return "Foo"

    def __exit__(self, type, value, trace):
        print("In __exit__()")


def get_sample():
    return Sample()


with get_sample() as sample:
    print("sample:", sample)
'''