#tf.trainable_variables()
import tensorflow as tf

starter_learning_rate = 0.01

x = tf.constant(3.0)
x2 = tf.constant(2.0)

y = tf.constant(19.0)#5x+2x2=15+4=19
w = tf.Variable(0.8,name = 'w')#5
w2 = tf.Variable(0.7, name = 'w2')#2
prediction = w * x + w2 * x2
loss = (prediction - y) ** 2

train_op = tf.train.GradientDescentOptimizer(starter_learning_rate).minimize(loss)

get_variables = tf.trainable_variables()
print(get_variables)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print('get_variables:',sess.run(get_variables))

    for i in range(1):
        print('predict:',sess.run(prediction))
        print('w:',sess.run(w))
        print('w2:',sess.run(w2))

        print(sess.run(loss))
        sess.run(train_op)