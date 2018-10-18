#this is a learning rate decay examples
import tensorflow as tf

starter_learning_rate = 0.01

x = tf.constant(5.0)
label = tf.constant(17.0)#3x+2
w = tf.Variable(0.8)
b = tf.Variable(0.1)
prediction = w * x + b
loss = (prediction - label) ** 2

train_op0 = tf.train.GradientDescentOptimizer(starter_learning_rate).minimize(loss)

steps_per_decay = 10
decay_factor = 0.96

global_step = tf.Variable(0, trainable=False)
learning_rate = tf.train.exponential_decay(learning_rate = starter_learning_rate,
                                           global_step = global_step,
                                           decay_steps = steps_per_decay,
                                           decay_rate = decay_factor,
                                           staircase = True,#If `True` decay the learning rate at discrete intervals
                                           #staircase = False,change learning rate at every step
                                           )


#passing global_step to minimize() will increment it at each step
train_op1 = (
    tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
)
#this cannot change lr
train_op2 = (
    #minimize()'s param global_step is the one to updates global_step
    #aka apply_gradients()
    tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
)

#tchange lr,but not use it.
train_op22 = (
    #minimize()'s param global_step is the one to updates global_step
    #aka apply_gradients()
    tf.train.GradientDescentOptimizer(starter_learning_rate).minimize(loss,global_step = global_step)
)
#this will change lr too
optimizer = tf.train.GradientDescentOptimizer(learning_rate)
grads = optimizer.compute_gradients(loss)
train_op3 = optimizer.apply_gradients(grads, global_step = global_step)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(30):
        print('global_step:',sess.run(global_step))
        print('predict:',sess.run(prediction))
        print('learning rate:',sess.run(learning_rate))
        print(sess.run(loss))
        # sess.run(train_op0)
        #sess.run(train_op1)
        #sess.run(train_op2)
        sess.run(train_op22)
        #sess.run(train_op3)


