import tensorflow as tf
import numpy as np
sess = tf.Session()

t_input = tf.placeholder(np.float32)
t2_input = tf.placeholder(np.float32,name = 'input')


print(type(t2_input))
print((t2_input))
print(sess.run(t_input,{t_input:33.33}))
print(sess.run(t2_input,{t2_input:133.33}))
print(tf.get_default_graph().get_operation_by_name('input'))
#print(sess.run(t2_input,{'input':133.33}))
#print(sess.run('input',{t2_input:133.33}))
#print(sess.run(input,{t2_input:133.33}))


