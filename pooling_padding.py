import tensorflow as tf
x = tf.constant([[1.,2.,3.],[4.,5.,6.]])
x = tf.reshape(x,[1,2,3,1])

print(tf.Session().run(x))

#valid_pad = tf.nn.conv2d(x, [1,2,2,1],[1,1,1,1], padding = 'VALID')
valid_pad = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding = 'VALID')
same_pad = tf.nn.max_pool(x,[1,2,2,1],[1,2,2,1],padding='SAME')
same_pad2 = tf.nn.max_pool(x,[1,2,2,1],[1,1,2,1],padding='SAME')

print(valid_pad.get_shape())
print(same_pad.get_shape())


print(tf.Session().run(valid_pad))
print(tf.Session().run(same_pad))
print(tf.Session().run(same_pad2))





