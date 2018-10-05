
import tensorflow as tf


#####################################################################################
#demo1:without x test

#define variable and error
'''
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

'''

####################################################################3
#demo2:select variable to train
#define variable and error
'''
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
'''

##########################################################################################3
#demo2.2  another way to collect var_list
'''
label = tf.constant(1,dtype = tf.float32)
x = tf.placeholder(dtype = tf.float32)
w1 = tf.Variable(4,dtype=tf.float32)
with tf.name_scope(name='selected_variable_to_trian'):
    w2 = tf.Variable(4,dtype=tf.float32)
w3 = tf.constant(4,dtype=tf.float32)

y_predict = w1*x+w2*x+w3*x

#define losses and train
make_up_loss = (y_predict - label)**3
optimizer = tf.train.GradientDescentOptimizer(0.01)

output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='selected_variable_to_trian')
train_step = optimizer.minimize(make_up_loss,var_list = output_vars)

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for _ in range(3000):
        w1_,w2_,w3_,loss_ = sess.run([w1,w2,w3,make_up_loss],feed_dict={x:1})
        print('variable is w1:',w1_,' w2:',w2_,' w3:',w3_, ' and the loss is ',loss_)
        sess.run(train_step,{x:1})
'''

##########################################################################################3
# #demo2.3  another way to avoid variable be train
#
# label = tf.constant(1,dtype = tf.float32)
# x = tf.placeholder(dtype = tf.float32)
# w1 = tf.Variable(4,dtype=tf.float32,trainable=False)
# w2 = tf.Variable(4,dtype=tf.float32)
# w3 = tf.constant(4,dtype=tf.float32)
#
# y_predict = w1*x+w2*x+w3*x
#
# #define losses and train
# make_up_loss = (y_predict - label)**3
# optimizer = tf.train.GradientDescentOptimizer(0.01)
#
# train_step = optimizer.minimize(make_up_loss)
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     for _ in range(3000):
#         w1_,w2_,w3_,loss_ = sess.run([w1,w2,w3,make_up_loss],feed_dict={x:1})
#         print('variable is w1:',w1_,' w2:',w2_,' w3:',w3_, ' and the loss is ',loss_)
#         sess.run(train_step,{x:1})




##########################################################################################3
#demo2.4  another way to avoid variable be train
#
# label = tf.constant(1,dtype = tf.float32)
# x = tf.placeholder(dtype = tf.float32)
# w1 = tf.Variable(4,dtype=tf.float32,trainable=False)
# w2 = tf.Variable(4,dtype=tf.float32)
# w3 = tf.constant(4,dtype=tf.float32)
#
# y_predict = w1*x+w2*x+w3*x
#
# #define losses and train
# make_up_loss = (y_predict - label)**3
# optimizer = tf.train.GradientDescentOptimizer(0.01)
#
# output_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
# train_step = optimizer.minimize(make_up_loss,var_list = output_vars)
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     for _ in range(3000):
#         w1_,w2_,w3_,loss_ = sess.run([w1,w2,w3,make_up_loss],feed_dict={x:1})
#         print('variable is w1:',w1_,' w2:',w2_,' w3:',w3_, ' and the loss is ',loss_)
#         sess.run(train_step,{x:1})


#################################################################
#demo2.5  combine of ompute_gradients() and apply_gradients()

# label = tf.constant(1,dtype = tf.float32)
# x = tf.placeholder(dtype = tf.float32)
# w1 = tf.Variable(4,dtype=tf.float32,trainable=False)
# w2 = tf.Variable(4,dtype=tf.float32)
# w3 = tf.Variable(4,dtype=tf.float32)
#
# y_predict = w1*x+w2*x+w3*x
#
# #define losses and train
# make_up_loss = (y_predict - label)**3
# optimizer = tf.train.GradientDescentOptimizer(0.01)
#
# w2_gradient = optimizer.compute_gradients(loss = make_up_loss, var_list = w2)
# train_step = optimizer.apply_gradients(grads_and_vars = (w2_gradient))
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     for _ in range(300):
#         w1_,w2_,w3_,loss_,w2_gradient_ = sess.run([w1,w2,w3,make_up_loss,w2_gradient],feed_dict={x:1})
#         print('variable is w1:',w1_,' w2:',w2_,' w3:',w3_, ' and the loss is ',loss_)
#         print('gradient:',w2_gradient_)
#         sess.run(train_step,{x:1})


###########################################################################################
#demo3:test all possible error formula
#define variable and error
'''
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
'''

###########################################################################################
#demo4:manual gradient descent in tensorflow
#y label
'''
y = tf.constant(3,dtype = tf.float32)
x = tf.placeholder(dtype = tf.float32)
w = tf.Variable(2,dtype=tf.float32)
#prediction
p = w*x

#define losses
l = tf.square(p - y)
g = tf.gradients(l, w)
learning_rate = tf.constant(1,dtype=tf.float32)
init = tf.global_variables_initializer()

#update
update = tf.assign(w, w - learning_rate * g[0])

with tf.Session() as sess:
    sess.run(init)
    print(sess.run([g,p,w], {x: 1}))
    for _ in range(5):
        w_,g_,l_ = sess.run([w,g,l],feed_dict={x:1})
        print('variable is w:',w_, ' g is ',g_,'  and the loss is ',l_)

        _ = sess.run(update,feed_dict={x:1})

'''

###########################################################################################
#demo4.2:todo:manual gradient descent in python api
#define variable and error
'''
from sympy import *
label = 1
x = 1
w = 4
learning_rate = 1
y_predict = w*x

#define losses and train
make_up_loss = tf.square(y_predict - label)
loss = Symbol('loss')
g = np.gradient(make_up_loss, w)
g = diff(make_up_loss,w)
print('g is ',g)
#update
def update():
    w = w - learning_rate * g * w

for _ in range(5):
    update()
    print('variable w is:', w, ' and the loss is ',make_up_loss)

'''

'''
from sympy import *
import numpy as np
w2=Symbol('w2')
l2 = 5*(w2**2)
deri = diff(l2,w2)#this is a expression,not a value
#print(type(float(deri)))
print('diff l2 w2:',deri)
#print('diff l2 w2:',deri(2))#error:'mul'object?
print('final der is ',deri * w2)
'''
'''
y = 3
x = 1
w_origin = 2
w=Symbol('w')
#prediction
p = w*x



#define losses
l = (p - y)**2
l_array = np.array([l])
print(l_array)
print(type(np.array(l)))
g = diff(l_array, w)
final_g = g * w_origin
learning_rate = 1

#update
for _ in range(5):

    print('variable is w:',w_origin, ' g is ',final_g,'  and the loss is ',l)

    w = w - learning_rate * final_g
    '''

###################################################################
#demo5 tensorflow momentum
'''
y = tf.constant(3,dtype = tf.float32)
x = tf.placeholder(dtype = tf.float32)
w = tf.Variable(2,dtype=tf.float32)
#prediction
p = w*x

#define losses
l = tf.square(p - y)
g = tf.gradients(l, w)
Mu = 0.8
LR = tf.constant(0.01,dtype=tf.float32)

init = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())

#update w
update = tf.train.MomentumOptimizer(LR, Mu).minimize(l)

with tf.Session() as sess:
    sess.run(init)
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())
    print(sess.run([g,p,w], {x: 1}))
    for _ in range(10):
        w_,g_,l_ = sess.run([w,g,l],feed_dict={x:1})
        print('variable is w:',w_, ' g is ',g_, '  and the loss is ',l_)

        sess.run([update],feed_dict={x:1})
'''
###########################################################################################
#demo5.2:manual momentum in tensorflow
'''
y = tf.constant(3,dtype = tf.float32)
x = tf.placeholder(dtype = tf.float32)
w = tf.Variable(2,dtype=tf.float32)
#prediction
p = w*x

#define losses
l = tf.square(p - y)
g = tf.gradients(l, w)
Mu = 0.8
LR = tf.constant(0.01,dtype=tf.float32)
#v = tf.Variable(0,tf.float32)#error?secend param is not dtype?
v = tf.Variable(0,dtype = tf.float32)
init = tf.global_variables_initializer()

#update w
update1 = tf.assign(v, Mu * v + g[0] * LR )
update2 = tf.assign(w, w - v)
#update = tf.group(update1,update2)#wrong sequence!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

with tf.Session() as sess:
    sess.run(init)
    print(sess.run([g,p,w], {x: 1}))
    for _ in range(10):
        w_,g_,l_,v_ = sess.run([w,g,l,v],feed_dict={x:1})
        print('variable is w:',w_, ' g is ',g_, ' v is ',v_,'  and the loss is ',l_)

        _ = sess.run([update1],feed_dict={x:1})
        _ = sess.run([update2],feed_dict={x:1})
'''



###########################################################################################
#demo6:adagrad optimizer in tensorflow
'''
y = tf.constant(3,dtype = tf.float32)
x = tf.placeholder(dtype = tf.float32)
w = tf.Variable(2,dtype=tf.float32)
#prediction
p = w*x

#define losses
l = tf.square(p - y)
g = tf.gradients(l, w)
LR = tf.constant(0.6,dtype=tf.float32)
optimizer = tf.train.AdagradOptimizer(LR)
update = optimizer.minimize(l)
init = tf.global_variables_initializer()


with tf.Session() as sess:
    sess.run(init)
    #print(sess.run([g,p,w], {x: 1}))
    for _ in range(20):
        w_,l_,g_ = sess.run([w,l,g],feed_dict={x:1})
        print('variable is w:',w_, 'g:',g_ ,'  and the loss is ',l_)

        _ = sess.run(update,feed_dict={x:1})

'''
###########################################################################################
#demo6.2:manual adagrad

#with tf.name_scope('initial'):
'''
y = tf.constant(3,dtype = tf.float32)
x = tf.placeholder(dtype=tf.float32)
w = tf.Variable(2,dtype=tf.float32,expected_shape=[1])
second_derivative = tf.Variable(0,dtype=tf.float32)
LR = tf.constant(0.6,dtype=tf.float32)
Regular = 1e-8

#prediction
p = w*x
#loss
l = tf.square(p - y)
#gradients
g = tf.gradients(l, w)
#print(g)
#print(tf.square(g))

#update
update1 = tf.assign_add(second_derivative,tf.square(g[0]))
g_final = LR * g[0] / (tf.sqrt(second_derivative) + Regular)
update2 = tf.assign(w, w - g_final)

#update = tf.assign(w, w - LR * g[0])

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    print(sess.run([g,p,w], {x: 1}))
    for _ in range(20):
        _ = sess.run(update1,feed_dict={x:1.0})
        w_,g_,l_,g_sec_ = sess.run([w,g,l,second_derivative],feed_dict={x:1.0})
        print('variable is w:',w_, ' g is ',g_,' g_sec_ is ',g_sec_,'  and the loss is ',l_)
        #sess.run(g_final)

        _ = sess.run(update2,feed_dict={x:1.0})

'''




#there is no SGD and mini-batch optimizer,just GD


