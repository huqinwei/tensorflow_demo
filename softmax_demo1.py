#https://blog.csdn.net/chengqiuming/article/details/80140928

import tensorflow as tf
labels = [[0,0,1],[0,1,0]]
labels_sparse = [2,1]
labels_without_one_hot = [[0.4,0.1,0.5],[0.3,0.6,0.1]]
logits = [[2,0.5,6],[0.1,0,3]]

logits_scaled = tf.nn.softmax(logits)
logits_scaled2 = tf.nn.softmax(logits_scaled)
logits_scaled2_2 = tf.nn.softmax(logits_scaled)
logits_scaled3 = tf.nn.softmax(logits_scaled2)
logits_scaled4 = tf.nn.softmax(logits_scaled3)

result1 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits)
result2 = tf.nn.softmax_cross_entropy_with_logits(labels=labels,logits=logits_scaled)
result3 = -tf.reduce_sum(labels*tf.log(logits_scaled),1)
result_without_one_hot = tf.nn.softmax_cross_entropy_with_logits(labels=labels_without_one_hot,logits=logits)
result_sparse = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels_sparse,logits=logits)

loss = tf.reduce_sum(result1)
loss2 = -tf.reduce_sum(labels*tf.log(logits_scaled))

with tf.Session() as sess:
	print("scaled=",sess.run(logits_scaled))
	print("scaled2=",sess.run(logits_scaled2))
	print("scaled2_2=",sess.run(logits_scaled2_2))
	print("scaled3=",sess.run(logits_scaled3))
	print("scaled4=",sess.run(logits_scaled4))

	print("result1=",sess.run(result1))
	print("result2=",sess.run(result2))
	print("result3=",sess.run(result3))
	print("result_without_one_hot=",sess.run(result_without_one_hot))
	print("result_sparse=",sess.run(result_sparse))

	print("loss=",sess.run(loss))
	print("loss2=",sess.run(loss2))





