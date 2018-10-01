import tensorflow as tf
#does tf.group have sequence????????????????????????

#demo1
#when range(500)two results randomly occur:499 and 500
#when range(3000)two results randomly occur:2999 and 3000
'''
with tf.name_scope('initial'):
    a = tf.Variable(0,dtype=tf.float32)
    b = tf.Variable(0,dtype=tf.float32)

#update
update1 = tf.assign_add(a,1)
update2 = tf.assign(b, a)
update = tf.group(update1,update2)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for _ in range(30000):
        _ = sess.run(update)
        print(sess.run(b))


'''
#demo2
#for _ in range(30000): output is randomly in 0,-1,-2,-3,-4   and the final output is -4
#for _ in range(30000): output is randomly in 0,1,2,3,4,5   and the final output is 4
#speculate wrong sequence:
#update1,update3,update2
#update2,update3,update1
'''
with tf.name_scope('initial'):
    a = tf.Variable(0,dtype=tf.float32)
    b = tf.Variable(0,dtype=tf.float32)

#update
update1 = tf.assign_add(a,1)
update2 = tf.assign_sub(a,1)
update3 = tf.assign(b, a)

update = tf.group(update1,update2,update3)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for _ in range(30000):
        _ = sess.run(update)
        print(sess.run(b))
'''
#demo3
#for _ in range(30000): output is randomly in 0,-1,-2,-3,-4   and the final output is -4
#for _ in range(30000): output is randomly in 0,1,2,3,4,5   and the final output is 4
#speculate wrong sequence:
#update1,update3,update2
#update2,update3,update1
with tf.name_scope('initial'):
    a = tf.Variable(0,dtype=tf.float32)
    b = tf.Variable(0,dtype=tf.float32)

#update
update1 = print('1')
update2 = print('2')
update3 = print('3')

update = tf.group(update1,update2,update3)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    for _ in range(30000):
        _ = sess.run(update)
        print(sess.run(b))

