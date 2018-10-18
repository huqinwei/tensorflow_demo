import tensorflow as tf
import numpy as np

x_data = np.random.rand(100).astype(np.float32)
y_data = x_data * 0.1 + 0.3


Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
print(Weights)
bias = tf.Variable(tf.zeros([1]))

predict = Weights*x_data + bias

loss = tf.reduce_mean(tf.square(predict-y_data))

optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()
print('init\'s type:',type(init))
print('init:',init)
print('(init):',(init))#same
sess = tf.Session()
sess.run(init)
for step in range(201):
	sess.run(train,{})
	if step % 20 == 0:
		Weights_, bias_, predict_ = sess.run([Weights, bias, predict])
		print(step,Weights_, bias_)
		print('x_data:', x_data[:5])
		print('y_data:', y_data[:5])
		print('predict:', predict_[:5])





