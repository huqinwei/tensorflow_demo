#imply auto-encoder by myself
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data


#data load
#mnist = input_data.read_datas('mnist/')
mnist = input_data.read_data_sets('mnist/', one_hot=False)
print(mnist.train.num_examples)
#print(len(mnist.train.data))
#train_image = mnist.train.data
#train_label = mnist.train.label


#hyper parameters
LR = 0.01
BATCH_SIZE = 256
TRAIN_EPOCHS = 5
DISPLAY_STEP = 1
examples_to_show = 10

#placeholder
input_x = tf.placeholder(dtype = tf.float32,shape=[None,784],name='input')

#variables:weights&bias
#784->256->128->256->784
weights11 = tf.Variable(tf.random_normal(shape = [784,256], dtype = tf.float32))
weights12 = tf.Variable(tf.random_normal(shape = [256,128], dtype = tf.float32))
weights21 = tf.Variable(tf.random_normal(shape = [128,256], dtype = tf.float32))
weights22 = tf.Variable(tf.random_normal(shape = [256,784], dtype = tf.float32))

biases11 = tf.Variable(tf.random_normal([256]))
biases12 = tf.Variable(tf.random_normal([128]))
biases21 = tf.Variable(tf.random_normal([256]))
biases22 = tf.Variable(tf.random_normal([784]))

#encoder
def encoder(x):
    l1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x,weights11),biases11))
    l2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(l1,weights12),biases12))
    return l2
#decoder
def decoder(x):
    l1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x,weights21),biases21))
    l2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(l1,weights22),biases22))
    return l2

#construct model
code = encoder(input_x)
output = decoder(code)
########################important########################################
#prediction
#Targets
#loss and optimizer                       
#wrong way:loss = tf.square(output - input_x)
y_true = input_x
y_pred = output
cost = tf.reduce_mean(tf.pow(y_true - y_pred,2))
########################################################
                       
#optimizer
optimizer = tf.train.AdamOptimizer(LR)
#train
train_step = optimizer.minimize(cost)
                       
#init  tf.group?local?
init = tf.global_variables_initializer()
#sess
with tf.Session() as sess:
    sess.run(init)
    #for i in range(TRAIN_EPOCHS):#mistake about EPOCH
    total_batch = int(mnist.train.num_examples / BATCH_SIZE)  #!!!!!!!!
    for epoch in range(TRAIN_EPOCHS):
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(BATCH_SIZE)
            sess.run(train_step,feed_dict = {input_x:batch_x})
            c = sess.run(cost,feed_dict = {input_x:batch_x})
        if epoch % DISPLAY_STEP == 0:
            print("epoch:",'%04d'%(epoch+1),'cost=',"{:.9f}".format(c))
    print("Optimization Finished")


    encode_decode = sess.run(y_pred,
                             feed_dict={input_x:mnist.test.images[:examples_to_show]})
    f, a = plt.subplots(2, 10, figsize=(10,2))
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i],(28,28)))
        a[1][i].imshow(np.reshape(encode_decode[i],(28,28)))
    plt.show()





        
 #       for i % 50 == 0:
            #accu = sess.run(accuracy,feed_dict = {input_x:mnist.test.data})
            #print('accuracy:',accu)#where did accuracy come from?
#            pre = sess.run(output,feed_dict = {input_x:mnist.test.data[:10]})
            #plot(pre.reshape(-1,28,28))













