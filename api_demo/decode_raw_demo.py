#decode_raw
import tensorflow as tf

str = tf.constant('hello world')
str2 = tf.constant('hello world!')

decoded = tf.decode_raw(str,tf.uint8)
#dc = tf.decode_raw(str,tf.uint16) #'hello world' is not multiple 2
decoded2 = tf.decode_raw(str2,tf.uint16)

sess = tf.Session()
print(sess.run(decoded))
print(sess.run(decoded2))
#[104 101 108 108 111  32 119 111 114 108 100]
#[25960 27756  8303 28535 27762  8548]
#pay attention on the order
#104 + 101  == 01101000 + 01100101== 26725
#101 + 104  == 25960
