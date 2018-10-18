import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp

W1 = tf.Variable([[1,2,3],[4,5,6]])#, name = 'variable1'
W2 = tf.Variable([[11,22,33],[44,55,66]])#, name = 'variable2'
print(W2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(W1))
print(sess.run(W2))
#save
saver = tf.train.Saver(var_list = {'variable1':W1, 'variable2':W2})
saver.save(sess,'./my_model_savedd')

################################################################################################
#reset graph and session,no need to delete code above.
tf.reset_default_graph()

W1 = tf.Variable([[1223,2,3],[433,5123,6]])
W2 = tf.Variable([[0,0,0],[0,0,0]])
sess = tf.Session()
#if all_tensors is True,print variable2 too
chkp.print_tensors_in_checkpoint_file('./my_model_savedd', tensor_name = 'variable1', all_tensors = False)#True
#load
saver = tf.train.Saver(var_list = {'variable1':W1})
Reverse = False
if Reverse == False:
    sess.run(tf.global_variables_initializer())#this is necessary
    saver.restore(sess,'./my_model_savedd')
else:
    saver.restore(sess, './my_model_savedd')
    sess.run(tf.global_variables_initializer())  # this is necessary

print('restored W1:\n', sess.run(W1))
print('initialized W2:\n', sess.run(W2))
