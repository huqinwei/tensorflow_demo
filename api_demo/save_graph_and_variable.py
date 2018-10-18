from tensorflow.python.platform import gfile
import tensorflow as tf
import os

W1 = tf.Variable([[1,2,3],[4,5,6]], name = 'var1')
W2 = tf.Variable([[11,22,33],[44,55,66]], name = 'var2')
print(W1)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

print(sess.run(W1))
print(sess.run(W2))

os.system('rm ./my_graph.pb')
tf.train.write_graph(sess.graph_def, './', 'my_graph.pb',False)

saver = tf.train.Saver(tf.global_variables())
os.system('rm ./my_model')
saver.save(sess,'./my_model',global_step = 0)

##############################################
tf.reset_default_graph()
###############################################
sess = tf.Session()#after reset_default_graph(),a new session is necessary

with gfile.FastGFile("./my_graph.pb", 'rb') as f:#"tmp/load/test.pb"
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  sess.graph.as_default()
  tf.import_graph_def(graph_def,name='')

W1 = sess.graph.get_tensor_by_name("var1:0")#you have to get a graph firstly
W200 = sess.graph.get_tensor_by_name("var2:0")
YYY = sess.graph.get_tensor_by_name("var1:0")
print('before:', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, W1)
tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, W200)
print('after:', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

try:
  saver = tf.train.Saver(tf.global_variables())  # 'Saver' misnomer! Better: Persister!
except:
  pass
print("load data")

saver.restore(sess,'./my_model-0')

print(W1.eval(session=sess))
print(W200.eval(session=sess))
print(YYY.eval(session=sess))
# print(sess.run(W2))
print(W1)
print(YYY)


