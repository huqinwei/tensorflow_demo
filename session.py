import tensorflow as tf

a = tf.constant([1.0,2.0],name='a')
b = tf.constant([1.0,2.0],name='b')
result = tf.add(a,b,name='add')



with tf.Session() as sess:
	print(sess.run(result))

sess = tf.Session()
with sess.as_default():
	print(result.eval())

sess = tf.Session()
print(sess.run(result))
#print(result.eval())#error:no default session
print(result.eval(session=sess))

sess = tf.InteractiveSession()
print(result.eval())#no error!!
sess.close()






