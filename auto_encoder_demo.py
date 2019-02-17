
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import numpy as np

tf.set_random_seed(1)

# Hyper Parameters
BATCH_SIZE = 64
LR = 0.002         # learning rate
CLR = 0.05         # classification learning rate
N_TEST_IMG = 5
IMG_SHOW_START = 5

# Mnist digits
mnist = input_data.read_data_sets('./mnist', one_hot=True)     # use not one-hotted target data
test_x = mnist.test.images[:200]
test_y = mnist.test.labels[:200]
print(test_y.shape)

# plot one example
print(mnist.train.images.shape)     # (55000, 28 * 28)
print(mnist.train.labels.shape)     # (55000, 10)
# plt.imshow(mnist.train.images[0].reshape((28, 28)), cmap='gray')
# plt.title('%i' % np.argmax(mnist.train.labels[0]))
# plt.show()

# tf placeholder
tf_x = tf.placeholder(tf.float32, [None, 28*28])    # value in the range of (0, 1)
tf_y = tf.placeholder(tf.float32, [None, 10])

# encoder
en0 = tf.layers.dense(tf_x, 128, tf.nn.tanh)
en1 = tf.layers.dense(en0, 64, tf.nn.tanh)
en2 = tf.layers.dense(en1, 12, tf.nn.tanh)
# encoded = tf.layers.dense(en2, 3)
encoded = tf.layers.dense(en2, 3)

# decoder
de0 = tf.layers.dense(encoded, 12, tf.nn.tanh)
de1 = tf.layers.dense(de0, 64, tf.nn.tanh)
de2 = tf.layers.dense(de1, 128, tf.nn.tanh)
decoded = tf.layers.dense(de2, 28*28, tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(labels=tf_x, predictions=decoded)
train = tf.train.AdamOptimizer(LR).minimize(loss)

#########################################################################################
#classification training
weights_fc1 = tf.Variable(initial_value = tf.truncated_normal([12,10],stddev=0.1),dtype=tf.float32)
biases_fc1 = tf.Variable(initial_value = tf.constant(0.1,shape = [10]))
weights_out = tf.Variable(initial_value = tf.truncated_normal([10,10],stddev=0.1),dtype=tf.float32)
biases_out = tf.Variable(initial_value = tf.constant(0.1,shape = [10]))

#fc1 = tf.nn.relu(tf.matmul(encoded, weights_fc1) + biases_fc1)#use layer encoded:0.754+
#fc1 = (tf.matmul(encoded, weights_fc1) + biases_fc1)#use layer encoded:0.63-
#fc1 = tf.nn.tanh(tf.matmul(encoded, weights_fc1) + biases_fc1)#use layer encoded:0.773+
fc1 = tf.nn.tanh(tf.matmul(en2,weights_fc1) + biases_fc1)#use layer en2:0.786+     #modify weights_fc1 correspondingly

pred = (tf.matmul(fc1, weights_out) + biases_out)#bad result in tanh
#pred = fc1#just use fc1 to predict  #use layer en2:0.719+ in 80000 steps,0.7639 in 800000 steps

pred_loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits = pred))
pred_train = tf.train.AdamOptimizer(CLR).minimize(pred_loss,var_list=[weights_fc1,weights_out,biases_fc1,biases_out])
pred_accuracy = tf.metrics.accuracy(labels = tf.argmax(tf_y,axis=1), predictions = tf.argmax(pred,axis=1))#dont forget axis=1
#########################################################################################
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())

# initialize figure
f, a = plt.subplots(2, N_TEST_IMG, figsize=(5, 2))
print(type(a))
print(a.shape)
plt.ion()   # continuously plot

# original data (first row) for viewing
view_data = mnist.test.images[IMG_SHOW_START : IMG_SHOW_START + N_TEST_IMG]
idx = 0#can not change to other!!!!!!!!!!!!!!!!!!!!!!this is axes,not data#this is fixed rows 0
for i in range(N_TEST_IMG):
    a[idx][i].imshow(np.reshape(view_data[i], (28, 28)), cmap='gray')
    a[idx][i].set_xticks(()); a[idx][i].set_yticks(())

for step in range(8000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _, encoded_, decoded_, loss_ = sess.run([train, encoded, decoded, loss], {tf_x: b_x})

    if step % 100 == 0:     # plotting
        print('train loss: %.4f' % loss_)
        #plotting decoded image (second row)
        decoded_data = sess.run(decoded, {tf_x: view_data})
        for i in range(N_TEST_IMG):
            a[1][i].clear()#this is fixed rows 1
            a[1][i].imshow(np.reshape(decoded_data[i], (28, 28)), cmap='gray')
            a[1][i].set_xticks(()); a[1][i].set_yticks(())
        plt.draw(); #plt.pause(0.01)
plt.ioff()

# visualize in 3D plot
view_data = test_x[:200]
encoded_data = sess.run(encoded, {tf_x: view_data})
fig = plt.figure(2); ax = Axes3D(fig)
X, Y, Z = encoded_data[:, 0], encoded_data[:, 1], encoded_data[:, 2]
for x, y, z, s in zip(X, Y, Z, np.argmax(test_y,axis = 1)):
    c = cm.rainbow(int(255*s/9)); ax.text(x, y, z, s, backgroundcolor=c)
ax.set_xlim(X.min(), X.max()); ax.set_ylim(Y.min(), Y.max()); ax.set_zlim(Z.min(), Z.max())
plt.show()

#
for step in range(80000):
    b_x, b_y = mnist.train.next_batch(BATCH_SIZE)
    _  = sess.run([pred_train], {tf_x: b_x, tf_y:b_y})

    if step % 100 == 0:     # plotting
        pred_, loss_, pred_loss_, accuracy_ = sess.run([ pred, loss, pred_loss, pred_accuracy],
                                                          {tf_x: test_x, tf_y: test_y})
        print('auto-encoder train loss: %.4f' % loss_)
        print('classification train loss: %.4f' % pred_loss_)
        print('classification train accuracy: %.4f' % accuracy_[1])
        #print('pred_: ' ,np.argmax(pred_,axis=1), ' label: ' ,np.argmax(test_y,axis=1))

plt.ioff()