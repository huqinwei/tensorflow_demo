#coding=utf-8
'''
生成三维数据,用平面去拟合
'''
import tensorflow as tf
import numpy as np

# 使用 NumPy 生成假数据(phony data), 总共 100 个点.
x_data = np.float32(np.random.rand(2, 10)) # 随机输入
x_data2 = np.float32((1,2,3,)) # 随机输入
x_data3 = np.float32(np.random.rand(3, 3)) # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300
#y_data = np.dot([1, 1], x_data) + 1.00
#y_data = np.dot([0.200, 0.200], x_data) + 0.300
print(x_data)
#print x_data2
#print x_data3
print(y_data)




# 构造一个线性模型
b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b

# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)

# 初始化变量
init = tf.global_variables_initializer()

# 启动图 (graph)
sess = tf.Session()
sess.run(init)

# 拟合平面
for step in range(0, 201):
    sess.run(train)
#    if step % 20 == 0:
#  	  print step, sess.run(W), sess.run(b)
# 得到最佳拟合结果 W: [[0.100 0.200]], b: [0.300]
