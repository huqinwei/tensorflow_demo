import tensorflow as tf

W1 = tf.Variable([[1,2,3],[4,5,6]])#, name = 'variable1'
W2 = tf.Variable([[11,22,33],[44,55,66]])#, name = 'variable2'
print(W2)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(W1))
print(sess.run(W2))
#save
saver = tf.train.Saver()
saver.save(sess,'./my_model_saved',global_step = 0)#global_step is necessary in practice,so i added it.
#saver.save(sess,'./my_model_saved')#ValueError1--solution2: dont use global_step!!!!!!!!!!!!!!!!!!!
#saver.save(sess,'./',global_step = 10000)#filename is different

################################################################################################
################################################################################################
#demo1
#reset graph and session,no need to delete code above.
tf.reset_default_graph()

W1 = tf.Variable([[1223,2,3],[433,5123,6]])
W2 = tf.Variable([[0,0,0],[0,0,0]])
print(W2)
sess = tf.Session()#after reset_default_graph(),a new session is necessary

#load
saver = tf.train.Saver()#after reset_default_graph(),a new saver is necessary
#saver.restore(sess,'./my_model_saved')#ValueError1--not a valid checkpoint
saver.restore(sess,'./my_model_saved-0')#ValueError1--solution1
#saver.restore(sess, './my_model_saved',global_step)#ValueError1--solution3--not support,no parameter named global_step

print(sess.run(W1))
print(sess.run(W2))

################################################################################################
#demo2
#reset graph and session,no need to delete code above.
# tf.reset_default_graph()
#
# W1 = tf.Variable([[1223,2,3],[433,5123,6]])
# W2 = tf.Variable([[0,0,0],[0,0,0]])
# print(W2)
# sess = tf.Session()#after reset_default_graph(),a new session is necessary
#
# #load
# saver = tf.train.Saver()#after reset_default_graph(),a new saver is necessary
# #saver.restore(sess,'./my_model_saved')#ValueError1--not a valid checkpoint
#
# saver.restore(sess,'./my_model_saved-0')#ValueError1--solution1
# #saver.restore(sess, './my_model_saved',global_step)#ValueError1--solution3--not support,no parameter named global_step
#
# print(sess.run(W1))
# print(sess.run(W2))



