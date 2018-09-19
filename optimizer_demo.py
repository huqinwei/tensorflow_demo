
import tensorflow as tf


#####################################################################################
#demo1:without x test

#define variable and error
label = tf.constant(1,dtype = tf.float32)
prediction_to_train = tf.Variable(3,dtype=tf.float32)

#define losses and train
manual_compute_loss = tf.square(prediction_to_train - label)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(manual_compute_loss)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(100):
        print('variable is ', sess.run(prediction_to_train), ' and the loss is ',sess.run(manual_compute_loss))
        sess.run(train_step)


####################################################################3
#demo2:select variable to train
#define variable and error
label = tf.constant(1,dtype = tf.float32)
x = tf.placeholder(dtype = tf.float32)
w1 = tf.Variable(4,dtype=tf.float32)
w2 = tf.Variable(4,dtype=tf.float32)
w3 = tf.constant(4,dtype=tf.float32)

y_predict = w1*x+w2*x+w3*x

#define losses and train
#make_up_loss = tf.square(y_predict - label)
#make_up_loss = (y_predict - label)**2
make_up_loss = (y_predict - label)**3
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(make_up_loss,var_list = w2)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        w1_,w2_,w3_,loss_ = sess.run([w1,w2,w3,make_up_loss],feed_dict={x:1})
        print('variable is w1:',w1_,' w2:',w2_,' w3:',w3_, ' and the loss is ',loss_)
        sess.run(train_step,{x:1})


###########################################################################################
#demo3:test all possible error formula
#define variable and error
label = tf.constant(1,dtype = tf.float32)
x = tf.placeholder(dtype = tf.float32)
w1 = tf.Variable(4,dtype=tf.float32)
w2 = tf.Variable(4,dtype=tf.float32)
w3 = tf.constant(4,dtype=tf.float32)

y_predict = w1*x+w2*x+w3*x

#define losses and train

#make_up_loss = tf.losses.sigmoid_cross_entropy(y_predict,label)
make_up_loss = tf.losses.mean_squared_error(y_predict,label)
#make_up_loss = (y_predict - label)**2
#make_up_loss = (y_predict - label)**3
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(make_up_loss,var_list = w2)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(300):
        w1_,w2_,w3_,loss_ = sess.run([w1,w2,w3,make_up_loss],feed_dict={x:1})
        print('variable is w1:',w1_,' w2:',w2_,' w3:',w3_, ' and the loss is ',loss_)
        sess.run(train_step,{x:1})













