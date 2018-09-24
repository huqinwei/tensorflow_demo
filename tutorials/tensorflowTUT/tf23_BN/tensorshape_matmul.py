import tensorflow as tf

c = tf.constant([[1.0,2.0,3.0],[4.0,5.0,6.0]])#2x3
d = tf.constant([[1.0,0.0],[0.0,1.0],[1.0,0.0],[0.0,1.0]])#4x2
d2 = tf.constant([[1.0,0.0,5.0],[0.0,1.0,4.0],[1.0,0.0,3.0],[0.0,1.0,2.0]])#4x3
print(c.shape)
print(d.shape)

#e = tf.matmul(c,d)#wrong shape:3!=4
f = tf.matmul(d,c)#change position4x2,2x3 ,then, direct matmul:4x3
print(f.shape)
g = tf.matmul(c,d,transpose_a=True,transpose_b=True)#2x3,4x2->3x2,2x4#success
#g2 = tf.matmul(c,d2,transpose_a=True,transpose_b=True)#2x3,4x3->3x2,3x4#failed
g22 = tf.matmul(c,d2,transpose_a=False,transpose_b=True)#2x3,4x3->2x3,3x4
print(g.shape)#transpose firstly:3x2,2x4,then matmul:3x4
print(g22.shape)#2x4
