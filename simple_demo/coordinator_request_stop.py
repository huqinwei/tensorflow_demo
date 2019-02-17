#try finally
import tensorflow as tf

array_to_outof = tf.constant([1,2,3,4,5,6,7,8])

counter = tf.Variable(0)
x = tf.Variable(1.)
w = tf.Variable(2.)

op = tf.multiply(w, x)
count_op = tf.assign_add(counter,1)

coord = tf.train.Coordinator()
sess = tf.Session()
with sess.as_default():
    sess.run(tf.global_variables_initializer())

    """Start Training"""
    threads = tf.train.start_queue_runners(coord=coord)
    try:
        while not coord.should_stop():
            print('in while-loop:')
            op_result_ = sess.run([op])
            print(op_result_)

            counter_ = sess.run(count_op)
            print('counter_:',counter_)
            array_to_outof[counter_]

    # except tf.errors.OutOfRangeError:
    #     print('except')
    except ValueError:
        print('ValueError exception!!!!!!!!!!!!!!!')
    finally:
        print('finally')
        coord.request_stop()#this is not necessary in this example to avoid dead-loop,just for reset?
    print('before join')
    coord.join(threads)
    print('after join')