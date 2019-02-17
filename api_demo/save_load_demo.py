import tensorflow as tf
import os
import numpy as np
from tensorflow.python.platform import gfile


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('summaries_dir', 'tmp/save_graph_logs', 'Summaries directory')

data = np.arange(10,dtype=np.int32)
with tf.Session() as sess:
  print("# build graph and run")
  input1= tf.placeholder(tf.int32, [10], name="input")
  output1= tf.add(input1, tf.constant(100,dtype=tf.int32), name="output") #  data depends on the input data
  saved_result= tf.Variable(data, name="saved_result")
  do_save=tf.assign(saved_result,output1)
  tf.initialize_all_variables()
  os.system("rm -rf tmp/save_graph_logs")
  merged = tf.summary.merge_all()
  train_writer = tf.summary.FileWriter(FLAGS.summaries_dir,
                                        sess.graph)

  os.system("rm -rf tmp/load")
  tf.train.write_graph(sess.graph_def, "tmp/load", "test.pb", False) #proto
  # now set the data:
  result,_=sess.run([output1,do_save], {input1: data}) # calculate output1 and assign to 'saved_result'
  saver = tf.train.Saver(tf.global_variables())
  saver.save(sess,"./checkpoint.data")

tf.reset_default_graph()#this is necessary,if you write this code in same file

with tf.Session() as persisted_sess:
  print("load graph")
  with gfile.FastGFile("tmp/load/test.pb", 'rb') as f:
      graph_def = tf.GraphDef()
      graph_def.ParseFromString(f.read())
      persisted_sess.graph.as_default()
      tf.import_graph_def(graph_def, name='')
  print("map variables")
  persisted_result = persisted_sess.graph.get_tensor_by_name("saved_result:0")
  print('before:', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
  tf.add_to_collection(tf.GraphKeys.GLOBAL_VARIABLES, persisted_result)
  print('after:', tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES))
  try:
      saver = tf.train.Saver(tf.global_variables())  # 'Saver' misnomer! Better: Persister!
  except:
      pass
  print("load data")
  saver.restore(persisted_sess, "./checkpoint.data")  # now OK
  print(persisted_result.eval())
  print("DONE")