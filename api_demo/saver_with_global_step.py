import tensorflow as tf

x = 10
y = 20
w = tf.Variable(0.)
prediction = w * x
loss = prediction - y


# Creates a variable to hold the global_step.
global_step_tensor = tf.Variable(10, dtype = tf.float32, trainable=False, name='global_step')

# Creates a session.
sess = tf.Session()
sess.run(tf.global_variables_initializer())
# Initializes the variable.
print('global_step: %s' % tf.train.global_step(sess, global_step_tensor))

print(tf.train.global_step(sess,global_step_tensor))
# print(tf.train.get_global_step('global_step:0'))
#global_step: 10
train = tf.train.GradientDescentOptimizer(0.01).minimize(loss,global_step= global_step_tensor)
saver = tf.train.Saver()

for i in range(3):
    sess.run(train)
    print(sess.run(global_step_tensor))
    saver.save(sess,'./my_train_too',global_step_tensor)

