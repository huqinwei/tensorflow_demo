import tensorflow as tf
import numpy
import math


a = tf.constant([10,20])
b = tf.constant([1.0, 2.0])

v = tf.Session().run(a)
print("v:",v)
v = tf.Session().run([a,b])
print("fetches can be a  list?? v:",v)

#MyData = collections.namedtuple('MyData',['a','b'])
#v = session.run({'k1':Mydata(a,b),'k2':[b,a]})
#print(v)


