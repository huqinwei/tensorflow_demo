import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

N_HIDDEN_UNITS=128
N_STEPS=28
N_INPUTS=28
N_CLASSES=10
BATCH_SIZE = 128
LR=0.001
training_iters = 100000


mnist = input_data.read_data_sets('mnist/',one_hot=True)#False?
#print(mnist.train.shape)
print(mnist.train.images.shape)
print(mnist.test.images.shape)

weights = {
    #28->128
    'in':tf.Variable(tf.random_normal([N_INPUTS,N_HIDDEN_UNITS])),
    #128->10
    'out':tf.Variable(tf.random_normal([N_HIDDEN_UNITS,N_CLASSES])),
                                              }
biases = {
    #28->128
    'in':tf.Variable(tf.constant(0.1, shape=[N_HIDDEN_UNITS,])),
    #128->10
    #shape=[-1,N_CLASSE] if tf.nn.bias_add????
    #wrong??
    'out':tf.Variable(tf.constant(0.1, shape=[N_CLASSES])),
    #'out':tf.Variable(tf.constant(0.1, shape=[N_CLASSES,])),
           }

def RNN(x,weights,biases):
    #x:128 batch,784inputs
    #inputs = tf.reshape(x,[-1,N_STEPS,N_INPUTS])
    #inputs:128 batch, 28 step, 28 inputs
    #weights:28,128,bias:128
    #inputs = tf.bias_add(tf.matmul(weights['in']),biases['in'])
    inputs = tf.reshape(x,[-1,N_INPUTS])
    inputs = tf.matmul(inputs,weights['in'])+biases['in']
    inputs = tf.reshape(inputs,[-1,N_STEPS,N_HIDDEN_UNITS])


    #cell i'm tf 1.10
    #rnn_cell = tf.nn.rnn_cell
    #wrong:is not tf.contrib.rnn and less paramas-----------------rnn_cell = tf.nn.rnn_cell.BasicRNNCell(N_HIDDEN_UNITS)
    #way 1:tf.nn.rnn_cell
    rnn_cell = tf.nn.rnn_cell.BasicLSTMCell(N_HIDDEN_UNITS,
                                            forget_bias=1.0,
                                            state_is_tuple=True)
    #way 2: tf__version__ >= 12:tf.contrib.rnn
    #rnn_cell = tf.contrib.rnn.BasicLSTMCell(N_HIDDEN_UNITS)
    initial_state = rnn_cell.zero_state(BATCH_SIZE, dtype=tf.float32)
    outputs,states = tf.nn.dynamic_rnn(rnn_cell,
                                      inputs,
                                      initial_state=initial_state,
                                       dtype=tf.float32,
                                       time_major=False)

    #28steps,128batch,10 outputs
    #way 1:outputs[-1]:128 batch 10 output
    print('\nbefore outputs:',outputs)
    outputs = tf.unstack(tf.transpose(outputs,[1,0,2]))
    print('\nafter outputs:',outputs)
    #print('shape:',outputs)
    final_outputs = outputs[-1]

    #way 2:states[1] == outputs[-1]
    #final_outputs = states[1]
    #128batch,10 outputs
    final_outputs = tf.matmul(final_outputs,weights['out']) + biases['out']

    return final_outputs

xs = tf.placeholder(tf.float32,[None,N_STEPS*N_INPUTS])
#xs = tf.placeholder(tf.float32,[None,N_STEPS,N_INPUTS])
ys = tf.placeholder(tf.float32,[None,N_CLASSES])

#128batch 10 output
outputs = RNN(xs,weights,biases)

#wrong!!!no need softmax------------prediction = tf.softmax(outputs)
#wrong:need average-------------------cost = tf.losses.softmax_cross_entropy(prediction,ys)
#128 batch,1 loss->1 average loss
#wrong api---------cost = tf.reduce_mean(tf.losses.softmax_cross_entropy(prediction,ys))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=outputs,labels=ys))
#wrong api---------------------train_step = tf.train.AdamGradientDescent().Optimizer.minize(cost)
train_step = tf.train.AdamOptimizer(LR).minimize(cost)

#128 batch,10 outputs, argmax axix=1
correct_pred = tf.equal(tf.argmax(outputs,axis=1),tf.argmax(ys,axis=1))
#128 batch,1 pred->1 accuracy
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))


#train
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)

    #why reshape?128 batch  784 pixel
    step=0
    while step*BATCH_SIZE < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(BATCH_SIZE)
##        #wrong???------batch_x = tf.reshape(batch_x,[-1,N_STEPS,N_INPUTS])
        #wrong!tf,ndarray------------batch_xs = tf.reshape(batch_xs,[BATCH_SIZE,N_STEPS,N_INPUTS])
        #print(type(batch_xs))
        #batch_xs = batch_xs.reshape([-1, N_STEPS, N_INPUTS])
        sess.run([train_step], feed_dict={
            xs: batch_xs,
            ys: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            xs: batch_xs,
            ys: batch_ys,
            }))
        step += 1




