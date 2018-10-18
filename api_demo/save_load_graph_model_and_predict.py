from tensorflow.python.platform import gfile
import tensorflow as tf
import os

Global_step = tf.Variable(0,name = 'global_step')

X = tf.constant(10.0)
Y = tf.constant(20.0)#Y=2*X
W = tf.Variable(1.1, name = 'weight')
Prediction = W*X
loss = (Y - Prediction)**2
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss,global_step = Global_step)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
saver = tf.train.Saver(tf.global_variables())

for i in range(100):
  print('Prediction:',sess.run(Prediction))
  print('loss:',sess.run(loss))
  print('W:',sess.run(W))
  sess.run(train_op)
  print(sess.run(Global_step))
  os.system('rm ./tmp2/my_graph.pb')
  tf.train.write_graph(sess.graph_def, './tmp2', 'my_graph.pb',False)

  # os.system('rm ./tmp2/my_model')
  saver.save(sess,'./tmp2/my_model',global_step = Global_step)



##############################################
tf.reset_default_graph()
###############################################
sess = tf.Session()#after reset_default_graph(),a new session is necessary

with gfile.FastGFile("./tmp2/my_graph.pb", 'rb') as f:#"tmp/load/test.pb"
  graph_def = tf.GraphDef()
  graph_def.ParseFromString(f.read())
  sess.graph.as_default()
  tf.import_graph_def(graph_def,name='')

X = tf.placeholder(dtype = tf.float32)
W = sess.graph.get_tensor_by_name("weight:0")#you have to get a graph firstly
Prediction = X*W

print('before:', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, W)
print('after:', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))

try:
  saver = tf.train.Saver(tf.global_variables())
except:
  pass
print("load data")

saver.restore(sess,'./tmp2/my_model-99')

print(W.eval(session=sess))
print(sess.run(Prediction,{X : 10.0}))
# print(sess.run(W2))


