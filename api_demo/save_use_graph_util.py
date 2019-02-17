from tensorflow.python.platform import gfile
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
from tensorflow.python.tools import inspect_checkpoint as chkp

with tf.variable_scope('net'):
  Global_step = tf.Variable(0,name = 'global_step')

  X = tf.placeholder(dtype = tf.float32, name = 'input')
  Y = tf.constant(20.0)#Y=2*X
  W = tf.Variable(1.1, name = 'weight')
  print(W)
  Prediction = tf.multiply(W,X,name = 'Prediction')
  loss = (Y - Prediction)**2
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss,global_step = Global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())

for i in range(100):
  # print('Prediction:',sess.run(Prediction, {X:10.0}))
  # print('loss:',sess.run(loss, {X:10.0}))
  # print('W:',sess.run(W))
  sess.run(train_op, {X:10.0})

output_graph_def = graph_util.convert_variables_to_constants(
        sess,
        sess.graph_def,
        'net/Prediction'.split(","),
)

# Write GraphDef to file if output path has been given.

with gfile.GFile('my_final_graph.pb', "wb") as f:
  f.write(output_graph_def.SerializeToString())

##############################################
tf.reset_default_graph()
###############################################
sess = tf.Session()#after reset_default_graph(),a new session is necessary
# with tf.variable_scope('import/net'):
# # with tf.variable_scope('import/net2'):
#   X = tf.placeholder(dtype = tf.float32, name = 'input')
#   Y = tf.constant(20.0)#Y=2*X
#   W = tf.Variable(1.1, name = 'weight')
#   print(W)
#   Prediction = tf.multiply(W,X,name = 'Prediction')

with gfile.FastGFile("my_final_graph.pb", 'rb') as f:#"tmp/load/test.pb"
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  sess.graph.as_default()
  tf.import_graph_def(graph_def,{})
# print('sess.graph:',sess.graph.get_operations())

Prediction = sess.graph.get_tensor_by_name("import/net/Prediction:0")#you have to get a graph firstly

# print('before:', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, Prediction)
# print('after:', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

print(sess.run(Prediction,{'import/net/input:0': 10.0}))
print(sess.run(Prediction,{'import/net/input:0': 15.0}))
tf.summary.FileWriter(logdir='import_net/',graph = sess.graph)


