# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

"""
Please note, this code is only for python 3+. If you are using python 2+, please modify the code accordingly.
"""
from __future__ import print_function
import tensorflow as tf
'''
state = tf.Variable(0, name='counter')
#print(state.name)
one = tf.constant(1)
two = tf.constant(2)

new_value = tf.add(state, two)
update = tf.assign(state,new_value)

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.initialize_all_variables()
else:
    init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        print(sess.run(update(state, new_value)))
        #sess.run(update(state, new_value))#has no effect
        #update.run()

        print(sess.run(state))
'''
#########################################################################################
#demo 2 calling class.func() to assign


class Test(object):
    def __init__(self):
        self.state = tf.Variable(0, name='counter')
        self.one = tf.constant(1)
        self.two = tf.constant(2)
    def update(self):
        tf.assign_add(self.state,self.two)

# tf.initialize_all_variables() no long valid from
# 2017-03-02 if using tensorflow >= 0.12
if int((tf.__version__).split('.')[1]) < 12 and int((tf.__version__).split('.')[0]) < 1:
    init = tf.group(tf.initialize_all_variables(), tf.local_variables_initializer())
else:
    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        #print(sess.run(update(state, new_value)))
        #sess.run(update(state, new_value))
        an_object = Test()
        an_object.update()

        print(sess.run(an_object.state))



