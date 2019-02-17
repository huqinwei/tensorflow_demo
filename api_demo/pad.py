import tensorflow as tf
with tf.Session() as sess:
  t = tf.constant([[1, 2, 3], [4, 5, 6]])

  paddings = tf.constant([[1, 1,], [2, 2]])

  # 'constant_values' is 0.
  # rank of 't' is 2.
  t1 = tf.pad(t, paddings, "CONSTANT")  # [[0, 0, 0, 0, 0, 0, 0],
                                   #  [0, 0, 1, 2, 3, 0, 0],
                                   #  [0, 0, 4, 5, 6, 0, 0],
                                   #  [0, 0, 0, 0, 0, 0, 0]]

  t2 = tf.pad(t, paddings, "REFLECT")  # [[6, 5, 4, 5, 6, 5, 4],
                                  #  [3, 2, 1, 2, 3, 2, 1],
                                  #  [6, 5, 4, 5, 6, 5, 4],
                                  #  [3, 2, 1, 2, 3, 2, 1]]

  t3 = tf.pad(t, paddings, "SYMMETRIC")  # [[2, 1, 1, 2, 3, 3, 2],
                                    #  [2, 1, 1, 2, 3, 3, 2],
                                    #  [5, 4, 4, 5, 6, 6, 5],
                                    #  [5, 4, 4, 5, 6, 6, 5]]

  print(paddings.eval())
  print(t.eval())
  print(t1.eval())
  print(t2.eval())
  print(t3.eval())