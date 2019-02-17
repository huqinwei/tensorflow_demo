import tensorflow as tf

sess = tf.Session()

x = tf.constant([[1.,2.,3.],[4.,5.,6.]])
x = tf.reshape(x,[1,2,3,1])
print(tf.Session().run(x))
print(x.shape)


#w_conv1 = tf.constant([2.,3.,1.,16.])
w_conv1 = tf.truncated_normal(([2,2,1,4]),stddev=0.1)
print('w_conv2\'s shape:', w_conv1.shape)

valid_pad = tf.nn.conv2d(x, w_conv1,[1,1,1,1],padding='VALID')
same_pad = tf.nn.conv2d(x, w_conv1,[1,1,1,1],padding='SAME')

print('valid_pad:\n', tf.Session().run(valid_pad))
print(valid_pad.get_shape())
print('same_pad:\n', tf.Session().run(same_pad))
print(same_pad.get_shape())












