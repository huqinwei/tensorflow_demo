#dependencies demo:use dependencies to run minimize()

# Apply gradients.
# apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
# with tf.control_dependencies([apply_gradient_op, variables_averages_op]):
#     train_op = tf.no_op(name='train')


import tensorflow as tf

starter_learning_rate = 0.01

x = tf.constant(5.0)
y = tf.constant(17.0)#3x+2
w = tf.Variable(0.8)
b = tf.Variable(0.1)

nonsense = tf.Variable(0)

prediction = w * x + b
loss = (prediction - y) ** 2

train_op = tf.train.GradientDescentOptimizer(starter_learning_rate).minimize(loss)

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

train_op2 = (
    tf.train.GradientDescentOptimizer(learning_rate).minimize(loss,global_step = global_step)
)

with tf.control_dependencies([train_op2]):
    train_op3 = tf.no_op(name = 'train')
    train_op4 = tf.assign(nonsense,1)


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(30):
        print('global_step:',sess.run(global_step))
        print('predict:',sess.run(prediction))
        print('learning rate:',sess.run(learning_rate))
        print(sess.run(loss))
        #sess.run(train_op)
        # sess.run(train_op2)
        # sess.run(train_op3)
        sess.run(train_op4)










