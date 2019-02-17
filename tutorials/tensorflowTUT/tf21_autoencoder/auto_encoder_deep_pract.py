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
LR = 0.001
BATCH_SIZE = 256
TRAIN_EPOCHS = 20
DISPLAY_STEP = 1
examples_to_show = 10


#variables:weights&bias
#784->256->128->256->784
n_input = 784
n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2
##n_hidden_3 = 32
##n_hidden_4 = 16
#placeholder
input_x = tf.placeholder(dtype = tf.float32,shape=[None,n_input],name='input')

weights = {
    'encoder_h1':tf.Variable(tf.random_normal([n_input,n_hidden_1], dtype = tf.float32)),
    'encoder_h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2], dtype = tf.float32)),
    'encoder_h3':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_3], dtype = tf.float32)),
    'encoder_h4':tf.Variable(tf.random_normal([n_hidden_3,n_hidden_4], dtype = tf.float32)),

    'decoder_h1':tf.Variable(tf.random_normal([n_hidden_4,n_hidden_3], dtype = tf.float32)),
    'decoder_h2':tf.Variable(tf.random_normal([n_hidden_3,n_hidden_2], dtype = tf.float32)),
    'decoder_h3':tf.Variable(tf.random_normal([n_hidden_2,n_hidden_1], dtype = tf.float32)),
    'decoder_h4':tf.Variable(tf.random_normal([n_hidden_1,n_input], dtype = tf.float32)),
    }
biases = {
    'encoder_b1':tf.Variable(tf.random_normal([n_hidden_1], dtype = tf.float32)),
    'encoder_b2':tf.Variable(tf.random_normal([n_hidden_2], dtype = tf.float32)),
    'encoder_b3':tf.Variable(tf.random_normal([n_hidden_3], dtype = tf.float32)),
    'encoder_b4':tf.Variable(tf.random_normal([n_hidden_4], dtype = tf.float32)),

    'decoder_b1':tf.Variable(tf.random_normal([n_hidden_3], dtype = tf.float32)),
    'decoder_b2':tf.Variable(tf.random_normal([n_hidden_2], dtype = tf.float32)),
    'decoder_b3':tf.Variable(tf.random_normal([n_hidden_1], dtype = tf.float32)),
    'decoder_b4':tf.Variable(tf.random_normal([n_input], dtype = tf.float32)),
    }

#encoder
def encoder(x):
    l1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x,weights['encoder_h1']),biases['encoder_b1']))
    l2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(l1,weights['encoder_h2']),biases['encoder_b2']))
    l3 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(l2,weights['encoder_h3']),biases['encoder_b3']))
    l4 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(l3,weights['encoder_h4']),biases['encoder_b4']))
    return l4
#decoder
def decoder(x):
    l1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(x,weights['decoder_h1']),biases['decoder_b1']))
    l2 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(l1,weights['decoder_h2']),biases['decoder_b2']))
    l3 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(l2,weights['decoder_h3']),biases['decoder_b3']))
    l4 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(l3,weights['decoder_h4']),biases['decoder_b4']))
    return l4

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


    encoder_result = sess.run(code,
                             feed_dict={input_x:mnist.test.images})
    plt.scatter(encoder_result[:,0],encoder_result[:,1],c=mnist.test.labels)
    plt.colorbar()
    plt.show()








