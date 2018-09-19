import tensorflow as tf
a = tf.constant([1.0,2.0],name='a')
b = tf.constant([1.0,2.0],name='b')
result = tf.add(a,b,name='add')
print(result)
print(result.get_shape)
print(result.shape)
