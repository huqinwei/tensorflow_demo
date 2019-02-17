import tensorflow as tf

a = tf.constant([1.0,2.0],name='a')
b = tf.constant([1.0,2.0],name='b')
c = tf.Variable(0)
result = tf.add(a,b,name='add')

#demo1
with tf.Session() as sess:
	print(sess.run(result))
	tf.global_variables_initializer().run()
	print(c.eval())
	sess_current = tf.get_default_session()
	print('use default session:', result.eval(session=sess_current))

	# RuntimeError: Attempted to use a closed Session.
	# sess_current.close()
	# sess_current = tf.get_default_session()
	# print('use default session:', result.eval(session=sess_current))

#demo2
sess = tf.Session()
with sess.as_default():
	print(result.eval())
	sess_current = tf.get_default_session()
	print('use default session:', result.eval(session=sess_current))

#demo3
sess = tf.Session()#this is not default session!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print(sess.run(result))
#print(result.eval())#error:no default session!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print(result.eval(session=sess))

#demo3.2
# ValueError: Cannot evaluate tensor using `eval()`: No default session is registered. Use `with sess.as_default()` or pass an explicit session to `eval(session=sess)`
# sess_current = tf.get_default_session()
# print('use default session:',result.eval(session = sess_current))
#demo3.3
# sess.as_default()#without with,not working!!!!!!!!!!!!!!!!!!!!!!!
# sess_current = tf.get_default_session()
# print('use default session:',result.eval(session = sess_current))

#demo4:this is default session

sess = tf.InteractiveSession()
print(result.eval())#no error!!
sess_current = tf.get_default_session()
print('use default session:', result.eval(session=sess_current))
sess.close()
#error:default session is closed
# sess_current = tf.get_default_session()
# print('use default session:', result.eval(session=sess_current))






