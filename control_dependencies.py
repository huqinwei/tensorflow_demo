import tensorflow as tf
#############################################
#exmaple1
'''
a = 5
b = tf.constant(value = 1)

##with tf.control_dependencies([a]):#can't convert int to tensor
##    print('hello world')

with tf.control_dependencies([b]):
    print('hello b')
'''
#############################################################
#example2
'''

x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x,1,name='x_plus_1')
with tf.control_dependencies([x_plus_1]):
    #y = x
    y = x_plus_1

def operation_a():
    #for i in range(5):#not working?
    
    with tf.control_dependencies([x_plus_1]):
        #y2 = x#wrong
        #y2 = x_plus_1
        y2 = x + 0.0
    return y2,x


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',graph = sess.graph)
    for i in range(5):
#        print('y:',sess.run(x,y))
        y_out,x_out = sess.run(operation_a())
        print('yout:',y_out)
        print('xout:',x_out)
'''


#########################################################3
#example3

x = tf.Variable(0.0)
x_plus_1 = tf.assign_add(x,1,name='x_plus_1')
with tf.control_dependencies([x_plus_1]):
    #y = x
    y = x_plus_1

def operation_a():
    #for i in range(5):#not working?
    
    with tf.control_dependencies([x_plus_1]):
#        y2 = x#wrong
        #y2 = x_plus_1
        y2 = x + 0.0
        #y2 = tf.identity(x)
    return y2,x

def operation_b():
    
    with tf.control_dependencies([x_plus_1]):
        y2 = x#wrong
    return y2,x


init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    writer = tf.summary.FileWriter('logs/',graph = sess.graph)
    for i in range(5):
#        print('y:',sess.run(x,y))
        y_out,x_out = sess.run(operation_a())
        print('yout:',y_out)
        print('xout:',x_out)
##        y_out2,x_out2 = sess.run(operation_b())
##        print('yout2:',y_out2)
##        print('xout2:',x_out2)
        
