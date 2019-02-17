import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

#hyparam
LR = 0.01
STEP_SIZE = 28
INPUT_SIZE = 28
BATCH_SIZE = 32
HIDDEN_CELL = 256
LSTM_LAYER = 3
CLASS_NUM = 10

#mnist load
MNIST = input_data.read_data_sets('../MNIST_data', one_hot = True)
print(MNIST.train.images.shape)
print(MNIST.train.labels.shape)
print(MNIST.test.labels.shape)

#def  network
tf_x = tf.placeholder(dtype = tf.float32, shape = [None,784],name = 'input')
tf_x_reshaped = tf.reshape(tf_x,[-1,STEP_SIZE,INPUT_SIZE])
tf_y = tf.placeholder(dtype = tf.float32, shape = [None,CLASS_NUM])
keep_prob = tf.placeholder(dtype = tf.float32)
print(tf_x)

def get_a_cell(i):
    lstm_cell =rnn.BasicLSTMCell(num_units=HIDDEN_CELL, forget_bias = 1.0, state_is_tuple = True, name = 'layer_%s'%i)
    print(type(lstm_cell))
    dropout_wrapped = rnn.DropoutWrapper(cell = lstm_cell, input_keep_prob = 1.0, output_keep_prob = keep_prob)
    return dropout_wrapped

multi_lstm = rnn.MultiRNNCell(cells = [get_a_cell(i) for i in range(LSTM_LAYER)],
                              state_is_tuple=True)#tf.nn.rnn_cell.MultiRNNCell

init_state = multi_lstm.zero_state(batch_size = BATCH_SIZE, dtype = tf.float32)

#example 1
print('a:',multi_lstm)
print('b:',tf_x_reshaped)
outputs, state = tf.nn.dynamic_rnn(multi_lstm, inputs = tf_x_reshaped, initial_state = init_state, time_major = False)
print('state:',state)
print('state[0]:',state[0])#layer 0's LSTMStateTuple
print('state[1]:',state[1])#layer 1's LSTMStateTuple
print('state[2]:',state[2])#layer 2's LSTMStateTuple
print('state[-1]:',state[-1])#layer 2's LSTMStateTuple
# print('state[1][31]:',state[1][31])#layer 1
print('outputs:', outputs)
print('outputs:', outputs[0])
print('outputs[31]:', outputs[31])
final_out = outputs[:,-1,:]
print(final_out)
h_state_0 = state[0][1]
h_state_1 = state[1][1]
h_state = state[-1][1]
h_state_2 = h_state
#example 2


#prediction and loss
W = tf.Variable(initial_value = tf.truncated_normal([HIDDEN_CELL, CLASS_NUM], stddev = 0.1 ), dtype = tf.float32)
print(W)
b = tf.Variable(initial_value = tf.constant(0.1, shape = [CLASS_NUM]), dtype = tf.float32)
predictions = tf.nn.softmax(tf.matmul(h_state, W) + b)
#sum   -ylogy^
cross_entropy = -tf.reduce_sum(tf_y * tf.log(predictions))
train_op = tf.train.AdamOptimizer(LR).minimize(cross_entropy)
#train
# keep_prob
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(2):
        x,y = MNIST.train.next_batch(BATCH_SIZE)
        _, loss_,outputs_, state_, h_state_0_, h_state_1_, h_state_2_ = \
            sess.run([train_op, cross_entropy,outputs, state, h_state_0, h_state_1, h_state_2], {tf_x:x, tf_y:y, keep_prob:1.0})
        print('loss:', loss_)
        print('outputs_:', outputs_.shape)
        # print('state_[0]:', state_[0].shape)
        # print('state_[1]:', state_[1].shape)
        print('h_state_0_:', h_state_0_.shape)
        print('h_state_1_:', h_state_1_.shape)
        print('h_state_2_:', h_state_2_.shape)
        print('h_state_2_ == outputs_[:,-1,:]:', h_state_2_ == outputs_[:,-1,:])














