import tensorflow as tf
sess = tf.Session()


t = tf.constant([1,2,3,4,5,6,7,8,9])
print(sess.run(t))

#wrong!!!print(sess.run(t.reshape([3,3])))
#wrong!!!print(sess.run(tf.reshape(t,3,3)))
print(sess.run(tf.reshape(t,[3,3])))

t2 = tf.constant(
    [
        [[1,1],[2,2]],
        [[3,3],[4,4]]
    ]
    )
print(t2)
print(t2.shape)#(2,2,2)
print(sess.run(t2))
print(sess.run(tf.reshape(t2,[2,4])))
print(sess.run(tf.reshape(t2,[4,2])))

t3 = tf.constant([[[1,1,1],
                   [2,2,2]],
                  [[3,3,3],
                   [4,4,4]],
                  [[5,5,5],
                   [6,6,6]]])
print(sess.run(t3))
print(t3.shape)
#print(sess.run(tf.reshape(t3,[0])))
print(sess.run(tf.reshape(t3,[-1])))
#flatten first,then reshape
print(sess.run(tf.reshape(t3,[2,9])))
print(sess.run(tf.reshape(t3,[-1,9,2])))
print(sess.run(tf.reshape(t3,[3,3,2])))
print(sess.run(tf.reshape(t3,[-1,2,3,3])))
print(sess.run(tf.reshape(t3,[2,3,3,-1])))
#wront!!!!!!!!!!!!print(sess.run(tf.reshape(t3,[-1,2,3,3,-1])))


print('#########################################')
t_input = tf.constant([
    [1,1,1,2,2,2],
                  [3,3,3,4,4,4],
                  [5,5,5,6,6,6]])

#input = 3*inputs=3*6
print(t_input.shape)


print(sess.run(t_input))
t_inputs = tf.reshape(t_input,[-1,3,2])
print('after')
print(sess.run(t_inputs))
print(t_input.shape)



